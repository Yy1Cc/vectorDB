#include "cluster/etcd_http_client.h"

#include <cstring>
#include "logger/logger.h"

namespace vectordb {

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  auto *str = static_cast<std::string *>(userp);
  size_t total = size * nmemb;
  str->append(static_cast<const char *>(contents), total);
  return total;
}

EtcdHttpClient::EtcdHttpClient(const std::string &endpoints) : base_url_(endpoints) {
  // etcd::SyncClient 接受 "http://127.0.0.1:2379" 格式
  // 去掉末尾的斜杠
  if (!base_url_.empty() && base_url_.back() == '/') {
    base_url_.pop_back();
  }
  curl_ = curl_easy_init();
}

EtcdHttpClient::~EtcdHttpClient() {
  if (curl_ != nullptr) {
    curl_easy_cleanup(curl_);
  }
}

auto EtcdHttpClient::base64Encode(const unsigned char *data, size_t len) -> std::string {
  BIO *b64 = BIO_new(BIO_f_base64());
  BIO *mem = BIO_new(BIO_s_mem());
  b64 = BIO_push(b64, mem);
  BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);  // 不换行
  BIO_write(b64, data, static_cast<int>(len));
  (void)BIO_flush(b64);
  BUF_MEM *bptr = nullptr;
  BIO_get_mem_ptr(b64, &bptr);
  std::string result(bptr->data, bptr->length);
  BIO_free_all(b64);
  return result;
}

auto EtcdHttpClient::base64Encode(const std::string &data) -> std::string {
  return base64Encode(reinterpret_cast<const unsigned char *>(data.data()), data.size());
}

auto EtcdHttpClient::base64Decode(const std::string &encoded) -> std::string {
  BIO *b64 = BIO_new(BIO_f_base64());
  BIO *mem = BIO_new_mem_buf(encoded.data(), static_cast<int>(encoded.size()));
  b64 = BIO_push(b64, mem);
  BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
  std::vector<char> buf(encoded.size());
  int len = BIO_read(b64, buf.data(), static_cast<int>(buf.size()));
  BIO_free_all(b64);
  if (len <= 0) {
    return {};
  }
  return {buf.data(), static_cast<size_t>(len)};
}

auto EtcdHttpClient::rangeEnd(const std::string &prefix) -> std::string {
  if (prefix.empty()) {
    // 空前缀 → range_end 为 \0 表示全范围
    return std::string(1, '\0');
  }
  std::string end = prefix;
  size_t i = end.size();
  while (i > 0) {
    auto &ch = end[i - 1];
    if (static_cast<unsigned char>(ch) != 0xFF) {
      ch = static_cast<char>(static_cast<unsigned char>(ch) + 1);
      end.resize(i);
      return end;
    }
    --i;
  }
  // 全部是 0xFF，range_end 用 \0 表示全范围
  return std::string(1, '\0');
}

auto EtcdHttpClient::httpPost(const std::string &path, const std::string &body) -> std::string {
  std::string url = base_url_ + path;
  std::string response;

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_reset(curl_);
  curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 10L);

  CURLcode res = curl_easy_perform(curl_);
  curl_slist_free_all(headers);

  if (res != CURLE_OK) {
    global_logger->error("etcd HTTP POST to {} failed: {}", url, curl_easy_strerror(res));
    return {};
  }

  long http_code = 0;
  curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code != 200) {
    global_logger->error("etcd HTTP POST to {} returned status {}", url, http_code);
    return {};
  }

  return response;
}

auto EtcdHttpClient::get(const std::string &key) -> EtcdResponse {
  EtcdResponse resp;
  std::string body = "{\"key\":\"" + base64Encode(key) + "\"}";
  std::string http_resp = httpPost("/v3/kv/range", body);

  if (http_resp.empty()) {
    resp.error_msg_ = "etcd HTTP request failed";
    return resp;
  }

  rapidjson::Document doc;
  doc.Parse(http_resp.c_str());

  if (doc.HasParseError() || !doc.IsObject()) {
    resp.error_msg_ = "etcd response parse error";
    return resp;
  }

  // etcd 错误响应 {"error":"...","code":N}
  if (doc.HasMember("error")) {
    resp.error_msg_ = doc["error"].IsString() ? doc["error"].GetString() : "unknown etcd error";
    return resp;
  }

  if (doc.HasMember("kvs") && doc["kvs"].IsArray() && !doc["kvs"].Empty()) {
    const auto &kv = doc["kvs"][0];
    if (kv.HasMember("value") && kv["value"].IsString()) {
      EtcdValue v;
      v.value_ = base64Decode(kv["value"].GetString());
      resp.values_.push_back(std::move(v));
      resp.keys_.push_back(key);
    }
  }

  resp.ok_ = true;
  return resp;
}

auto EtcdHttpClient::ls(const std::string &prefix) -> EtcdResponse {
  EtcdResponse resp;
  std::string end = rangeEnd(prefix);
  std::string body = "{\"key\":\"" + base64Encode(prefix) + "\",\"range_end\":\"" + base64Encode(end) + "\"}";
  std::string http_resp = httpPost("/v3/kv/range", body);

  if (http_resp.empty()) {
    resp.error_msg_ = "etcd HTTP request failed";
    return resp;
  }

  rapidjson::Document doc;
  doc.Parse(http_resp.c_str());

  if (doc.HasParseError() || !doc.IsObject()) {
    resp.error_msg_ = "etcd response parse error";
    return resp;
  }

  if (doc.HasMember("error")) {
    resp.error_msg_ = doc["error"].IsString() ? doc["error"].GetString() : "unknown etcd error";
    return resp;
  }

  if (doc.HasMember("kvs") && doc["kvs"].IsArray()) {
    for (const auto &kv : doc["kvs"].GetArray()) {
      std::string k;
      std::string val;
      if (kv.HasMember("key") && kv["key"].IsString()) {
        k = base64Decode(kv["key"].GetString());
      }
      if (kv.HasMember("value") && kv["value"].IsString()) {
        val = base64Decode(kv["value"].GetString());
      }
      resp.keys_.push_back(std::move(k));
      resp.values_.push_back({std::move(val)});
    }
  }

  resp.ok_ = true;
  return resp;
}

auto EtcdHttpClient::set(const std::string &key, const std::string &value) -> EtcdResponse {
  EtcdResponse resp;
  std::string body = "{\"key\":\"" + base64Encode(key) + "\",\"value\":\"" + base64Encode(value) + "\"}";
  std::string http_resp = httpPost("/v3/kv/put", body);

  if (http_resp.empty()) {
    resp.error_msg_ = "etcd HTTP request failed";
    return resp;
  }

  rapidjson::Document doc;
  doc.Parse(http_resp.c_str());

  if (doc.HasMember("error")) {
    resp.error_msg_ = doc["error"].IsString() ? doc["error"].GetString() : "unknown etcd error";
    return resp;
  }

  resp.ok_ = true;
  return resp;
}

auto EtcdHttpClient::rm(const std::string &key) -> EtcdResponse {
  EtcdResponse resp;
  std::string body = "{\"key\":\"" + base64Encode(key) + "\"}";
  std::string http_resp = httpPost("/v3/kv/deleterange", body);

  if (http_resp.empty()) {
    resp.error_msg_ = "etcd HTTP request failed";
    return resp;
  }

  rapidjson::Document doc;
  doc.Parse(http_resp.c_str());

  if (doc.HasMember("error")) {
    resp.error_msg_ = doc["error"].IsString() ? doc["error"].GetString() : "unknown etcd error";
    return resp;
  }

  resp.ok_ = true;
  return resp;
}

}  // namespace vectordb
