#pragma once

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <string>
#include <vector>

namespace vectordb {

// 兼容 etcd::Value 的轻量封装
struct EtcdValue {
  std::string value_;
  auto as_string() const -> const std::string & { return value_; }
};

// 兼容 etcd::Response 的轻量封装
struct EtcdResponse {
  bool ok_ = false;
  std::string error_msg_;
  std::vector<std::string> keys_;
  std::vector<EtcdValue> values_;

  auto is_ok() const -> bool { return ok_; }
  auto error_message() const -> const std::string & { return error_msg_; }
  auto value() const -> EtcdValue {
    return values_.empty() ? EtcdValue{} : values_[0];
  }
  auto keys() const -> const std::vector<std::string> & { return keys_; }
  auto values() const -> const std::vector<EtcdValue> & { return values_; }
};

// 通过 etcd v3 HTTP/gRPC-gateway API 访问 etcd，避免 protobuf/gRPC 版本冲突
class EtcdHttpClient {
 public:
  explicit EtcdHttpClient(const std::string &endpoints);
  ~EtcdHttpClient();

  auto get(const std::string &key) -> EtcdResponse;
  auto ls(const std::string &prefix) -> EtcdResponse;
  auto set(const std::string &key, const std::string &value) -> EtcdResponse;
  auto rm(const std::string &key) -> EtcdResponse;

 private:
  auto httpPost(const std::string &path, const std::string &body) -> std::string;
  static auto base64Encode(const unsigned char *data, size_t len) -> std::string;
  static auto base64Encode(const std::string &data) -> std::string;
  static auto base64Decode(const std::string &encoded) -> std::string;
  // 计算前缀的 range_end（用于范围查询/删除）
  static auto rangeEnd(const std::string &prefix) -> std::string;

  std::string base_url_;
  CURL *curl_ = nullptr;
};

}  // namespace vectordb
