#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include "common/vector_cfg.h"
#include "common/vector_init.h"
#include "httpserver/http_server.h"
#include "index/index_factory.h"
#include "logger/logger.h"

namespace {

// NOLINTNEXTLINE
auto WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) -> size_t {
  auto *str = static_cast<std::string *>(userp);
  size_t total = size * nmemb;
  str->append(static_cast<const char *>(contents), total);
  return total;
}

// 向 Master 上报自身信息（一次尝试）。成功返回 true。
auto RegisterToMasterOnce() -> bool {
  const auto &master_address = vectordb::Cfg::Instance().MasterAddress();
  int master_port = vectordb::Cfg::Instance().MasterPort();
  if (master_address.empty() || master_port == 0) {
    // 未配置 Master：沿用人工注册方式，不重试
    return true;
  }

  std::string url =
      "http://" + master_address + ":" + std::to_string(master_port) + "/MasterService/AddNode";

  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();
  std::string node_url = vectordb::Cfg::Instance().NodeUrl();
  std::string endpoint = vectordb::Cfg::Instance().RaftEndpoint();

  doc.AddMember("instanceId", vectordb::Cfg::Instance().InstanceId(), allocator);
  doc.AddMember("nodeId", static_cast<uint64_t>(vectordb::Cfg::Instance().RaftNodeId()), allocator);
  doc.AddMember("url", rapidjson::Value(node_url.c_str(), allocator), allocator);
  doc.AddMember("endpoint", rapidjson::Value(endpoint.c_str(), allocator), allocator);
  doc.AddMember("status", 1, allocator);
  doc.AddMember("role", 1, allocator);  // 先按 follower 登记，真实角色由 Master 探测纠正

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    vectordb::global_logger->error("RegisterToMaster: curl init failed");
    return false;
  }

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  std::string response_str;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.GetString());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

  CURLcode res = curl_easy_perform(curl);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    vectordb::global_logger->warn("RegisterToMaster: request to {} failed", url);
    return false;
  }

  rapidjson::Document resp;
  resp.Parse(response_str.c_str());
  if (resp.HasParseError() || !resp.IsObject() || !resp.HasMember("retCode")) {
    vectordb::global_logger->warn("RegisterToMaster: unexpected response {}", response_str);
    return false;
  }
  return resp["retCode"].GetInt() == 0;
}

// 后台注册：Master 可能尚未启动，注册失败不应阻塞本节点提供服务。
// 每 5 秒重试一次，最多约 2 分钟。
void StartRegisterToMaster() {
  std::thread([]() {
    const int kMaxAttempts = 24;
    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
      if (RegisterToMasterOnce()) {
        vectordb::global_logger->info("Register to master succeeded (attempt {})", attempt);
        return;
      }
      vectordb::global_logger->warn("Register to master failed (attempt {}/{}), retry in 5s", attempt,
                                    kMaxAttempts);
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    vectordb::global_logger->error("Give up registering to master after {} attempts", kMaxAttempts);
  })
      .detach();
}

}  // namespace

// NOLINTNEXTLINE
auto main(int argc, char *argv[]) -> int {
  int node_id = 1;
  if (argc == 2) {
    node_id = std::atoi(argv[1]);  // Convert argument to integer if provided
    std::cout << "The node_id you input is " << node_id << std::endl;
  } else {
    std::cout << "No number provided, using default value: " << node_id << std::endl;
  }
  vectordb::VdbServerInit(node_id);
  vectordb::global_logger->info("Global IndexFactory initialized");

  // 创建并启动HTTP服务器

  vectordb::VectorDatabase vector_database(vectordb::Cfg::Instance().RocksDbPath(),
                                           vectordb::Cfg::Instance().WalPath());
  vector_database.ReloadDatabase();
  vectordb::global_logger->info("VectorDatabase initialized");

  int raft_node_id = vectordb::Cfg::Instance().RaftNodeId();
  std::string endpoint = vectordb::Cfg::Instance().RaftEndpoint();
  int raft_port = vectordb::Cfg::Instance().RaftPort();

  vectordb::RaftStuff raft_stuff(raft_node_id, endpoint, raft_port, &vector_database);
  vectordb::global_logger->info("RaftStuff object created with node_id: {}, endpoint: {}, port: {}", raft_node_id, endpoint,
                                raft_port);  // 添加调试日志

  vectordb::HttpServer server;
  server.Init(&vector_database, &raft_stuff);
  vectordb::global_logger->info("HttpServer created");

  std::string server_addr =
      vectordb::Cfg::Instance().Address() + ":" + std::to_string(vectordb::Cfg::Instance().Port());
  LOG(INFO) << "listen at:" << server_addr;

  brpc::ServerOptions options;
  if (server.Start(server_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to start server";
    return -1;
  }

  // 服务已就绪后再向 Master 注册，保证 Master 探测时本节点已可应答
  StartRegisterToMaster();

  server.RunUntilAskedToQuit();

  return 0;
}
