#include "httpserver/master_service_impl.h"
#include <cctype>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include "common/constants.h"
namespace vectordb {

namespace {

// 从 etcd key 末段解析数字，如 "/instances/1/nodes/2" -> 2；解析失败返回 0
auto ParseTrailingId(const std::string &key) -> uint64_t {
  auto pos = key.rfind('/');
  if (pos == std::string::npos) {
    return 0;
  }
  try {
    return std::stoull(key.substr(pos + 1));
  } catch (const std::exception &) {
    return 0;
  }
}

// 从 "/instances/{instanceId}/nodes/{nodeId}" 解析出 instanceId；解析失败返回 0
auto ParseNodeKeyInstanceId(const std::string &key) -> uint64_t {
  const std::string prefix = "/instances/";
  if (key.rfind(prefix, 0) != 0) {
    return 0;
  }
  auto pos = key.find('/', prefix.size());
  if (pos == std::string::npos) {
    return 0;
  }
  try {
    return std::stoull(key.substr(prefix.size(), pos - prefix.size()));
  } catch (const std::exception &) {
    return 0;
  }
}

// 从 "/instancesConfig/{instanceId}/partitionConfig" 解析出 instanceId；解析失败返回 0
auto ParseConfigInstanceId(const std::string &key) -> uint64_t {
  const std::string prefix = "/instancesConfig/";
  if (key.rfind(prefix, 0) != 0) {
    return 0;
  }
  auto pos = key.find('/', prefix.size());
  if (pos == std::string::npos) {
    return 0;
  }
  try {
    return std::stoull(key.substr(prefix.size(), pos - prefix.size()));
  } catch (const std::exception &) {
    return 0;
  }
}

// 判断某节点是否已追平 leader：落后不超过阈值即视为可接管分区
auto HasCaughtUp(uint64_t node_log_idx, uint64_t leader_log_idx) -> bool {
  if (node_log_idx >= leader_log_idx) {
    return true;
  }
  return (leader_log_idx - node_log_idx) <= kMaxLogLagForReady;
}

}  // namespace

auto ServerInfo::FromJson(const rapidjson::Document &value) -> ServerInfo {
  ServerInfo info;
  info.url_ = value["url"].GetString();
  info.role_ = static_cast<ServerRole>(value["role"].GetInt());
  return info;
}

auto ServerInfo::ToJson() const -> rapidjson::Document {
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  // 将 url 和 role 转换为 rapidjson::Value
  rapidjson::Value url_value;
  url_value.SetString(url_.c_str(), allocator);

  rapidjson::Value role_value;
  role_value.SetInt(static_cast<int>(role_));

  // 使用正确的类型添加成员
  doc.AddMember("url", url_value, allocator);
  doc.AddMember("role", role_value, allocator);

  return doc;
}

void MasterServiceImpl::GetNodeInfo(::google::protobuf::RpcController *controller,
                                    const ::nvm::HttpRequest * /*request*/, ::nvm::HttpResponse * /*response*/,
                                    ::google::protobuf::Closure *done) {
  global_logger->info("Received GetNodeInfo request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 从JSON请求中获取ID
  uint64_t instance_id = json_request[INSTANCE_ID].GetUint64();
  uint64_t node_id = json_request[NODE_ID].GetUint64();

  try {
    std::string etcd_key = "/instances/" + std::to_string(instance_id) + "/nodes/" + std::to_string(node_id);
    EtcdResponse etcd_response = etcd_client_.get(etcd_key);
    if (!etcd_response.is_ok()) {
      global_logger->error("etcd response fail");
      SetResponse(cntl, 1, "Error accessing etcd: " + etcd_response.error_message());
      return;
    }

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    // 解析节点信息
    rapidjson::Document node_doc;
    node_doc.Parse(etcd_response.value().as_string().c_str());
    if (!node_doc.IsObject()) {
      SetResponse(cntl, 1, "Invalid JSON format");
      return;
    }

    // 构建响应
    doc.AddMember("instanceId", instance_id, allocator);
    doc.AddMember("nodeId", node_id, allocator);
    doc.AddMember("nodeInfo", node_doc, allocator);

    SetResponse(cntl, 0, "Node info retrieved successfully", &doc);
  } catch (const std::exception &e) {
    global_logger->error("GetNodeInfo exception");
    SetResponse(cntl, 1, "Exception accessing etcd: " + std::string(e.what()));
  }
}

void MasterServiceImpl::AddNode(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                                ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->info("Received AddNode request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  try {
    uint64_t instance_id = json_request[INSTANCE_ID].GetUint64();
    uint64_t node_id = json_request[NODE_ID].GetUint64();
    std::string etcd_key = "/instances/" + std::to_string(instance_id) + "/nodes/" + std::to_string(node_id);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json_request.Accept(writer);

    etcd_client_.set(etcd_key, buffer.GetString());
    SetResponse(cntl, 0, "Node added successfully");
  } catch (const std::exception &e) {
    global_logger->error("AddNode exception");
    SetResponse(cntl, 1, std::string("Error accessing etcd: ") + e.what());
  }
}

void MasterServiceImpl::RemoveNode(::google::protobuf::RpcController *controller,
                                   const ::nvm::HttpRequest *
                                   /*request*/,
                                   ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->info("Received RemoveNode request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 从JSON请求中获取ID
  uint64_t instance_id = json_request[INSTANCE_ID].GetUint64();
  uint64_t node_id = json_request[NODE_ID].GetUint64();

  std::string etcd_key = "/instances/" + std::to_string(instance_id) + "/nodes/" + std::to_string(node_id);

  try {
    EtcdResponse etcd_response = etcd_client_.rm(etcd_key);
    if (!etcd_response.is_ok()) {
      global_logger->error("RemoveNode etcd error");
      SetResponse(cntl, 1, "Error removing node from etcd: " + etcd_response.error_message());
      return;
    }
    SetResponse(cntl, 0, "Node removed successfully");
  } catch (const std::exception &e) {
    global_logger->error("RemoveNode etcd Exception");
    SetResponse(cntl, 1, "Exception accessing etcd: " + std::string(e.what()));
  }
}

void MasterServiceImpl::GetInstance(::google::protobuf::RpcController *controller,
                                    const ::nvm::HttpRequest *
                                    /*request*/,
                                    ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->info("Received GetInstance request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 从JSON请求中获取ID
  uint64_t instance_id = json_request[INSTANCE_ID].GetUint64();
  try {
    std::string etcd_key_prefix = "/instances/" + std::to_string(instance_id) + "/nodes/";
    global_logger->debug("etcd key prefix: {}", etcd_key_prefix);

    EtcdResponse etcd_response = etcd_client_.ls(etcd_key_prefix);
    global_logger->debug("etcd ls response received");

    if (!etcd_response.is_ok()) {
      global_logger->error("Error accessing etcd: {}", etcd_response.error_message());
      SetResponse(cntl, 1, "Error accessing etcd: " + etcd_response.error_message());
      return;
    }

    const auto &keys = etcd_response.keys();
    const auto &values = etcd_response.values();

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    rapidjson::Value nodes_array(rapidjson::kArrayType);
    for (size_t i = 0; i < keys.size(); ++i) {
      global_logger->debug("Processing key: {}", keys[i]);
      rapidjson::Document node_doc;
      node_doc.Parse(values[i].as_string().c_str());
      if (!node_doc.IsObject()) {
        global_logger->warn("Invalid JSON format for key: {}", keys[i]);
        continue;
      }

      // 使用 CopyFrom 方法将节点信息添加到数组中
      rapidjson::Value node_value(node_doc, allocator);
      nodes_array.PushBack(node_value, allocator);
    }

    doc.AddMember("instanceId", instance_id, allocator);
    doc.AddMember("nodes", nodes_array, allocator);

    global_logger->info("Instance info retrieved successfully for instanceId: {}", instance_id);
    SetResponse(cntl, 0, "Instance info retrieved successfully", &doc);
  } catch (const std::exception &e) {
    global_logger->error("Exception accessing etcd: {}", e.what());
    SetResponse(cntl, 1, "Exception accessing etcd: " + std::string(e.what()));
  }
}

void MasterServiceImpl::UpdateNodeStates() {
  CURL *curl = curl_easy_init();

  if (curl == nullptr) {
    global_logger->error("CURL initialization failed");
    return;
  }

  try {
    std::string nodes_key_prefix = "/instances/";
    global_logger->info("Fetching nodes list from etcd");
    EtcdResponse etcd_response = etcd_client_.ls(nodes_key_prefix);

    for (size_t i = 0; i < etcd_response.keys().size(); ++i) {
      const std::string &node_key = etcd_response.keys()[i];
      const std::string &node_value = etcd_response.values()[i].as_string();

      rapidjson::Document node_doc;
      node_doc.Parse(node_value.c_str());
      if (!node_doc.IsObject()) {
        global_logger->warn("Invalid JSON format for node: {}", node_key);
        continue;
      }

      std::string get_node_url = std::string(node_doc["url"].GetString()) + "/AdminService/GetNode";
      global_logger->debug("Sending request to {}", get_node_url);

      curl_easy_setopt(curl, CURLOPT_URL, get_node_url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, BaseServiceImpl::WriteCallback);

      std::string response_str;
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);

      CURLcode res = curl_easy_perform(curl);
      bool needs_update = false;

      if (res != CURLE_OK) {
        global_logger->error("curl_easy_perform() failed: {}", curl_easy_strerror(res));
        node_error_counts_[node_key]++;
        if (node_error_counts_[node_key] >= 5 && node_doc["status"].GetInt() != 0) {
          node_doc["status"].SetInt(0);  // Set status to 0 (abnormal)
          needs_update = true;
        }
      } else {
        node_error_counts_[node_key] = 0;  // Reset error count
        if (node_doc["status"].GetInt() != 1) {
          node_doc["status"].SetInt(1);  // Set status to 1 (normal)
          needs_update = true;
        }

        rapidjson::Document get_node_response;
        get_node_response.Parse(response_str.c_str());
        if (get_node_response.HasMember("node") && get_node_response["node"].IsObject()) {
          std::string state = get_node_response["node"]["state"].GetString();
          int new_role = (state == "leader") ? 0 : 1;

          if (node_doc["role"].GetInt() != new_role) {
            node_doc["role"].SetInt(new_role);  // Update role
            needs_update = true;
          }

          // 采集负载指标（老版本节点不返回这两个字段，容错跳过）
          const auto &node_obj = get_node_response["node"];
          if (node_obj.HasMember("vectorCount") && node_obj["vectorCount"].IsUint64()) {
            auto new_vector_count = node_obj["vectorCount"].GetUint64();
            if (!node_doc.HasMember("vectorCount") || !node_doc["vectorCount"].IsUint64()) {
              // 首次采集，字段尚不存在（RapidJSON 对不存在的 key 用 operator[] 会断言）
              node_doc.AddMember("vectorCount", rapidjson::Value(new_vector_count), node_doc.GetAllocator());
              needs_update = true;
            } else if (node_doc["vectorCount"].GetUint64() != new_vector_count) {
              node_doc["vectorCount"].SetUint64(new_vector_count);
              needs_update = true;
            }
          }
          if (node_obj.HasMember("diskUsageBytes") && node_obj["diskUsageBytes"].IsUint64()) {
            auto new_disk_usage = node_obj["diskUsageBytes"].GetUint64();
            if (!node_doc.HasMember("diskUsageBytes") || !node_doc["diskUsageBytes"].IsUint64()) {
              node_doc.AddMember("diskUsageBytes", rapidjson::Value(new_disk_usage), node_doc.GetAllocator());
              needs_update = true;
            } else if (node_doc["diskUsageBytes"].GetUint64() != new_disk_usage) {
              node_doc["diskUsageBytes"].SetUint64(new_disk_usage);
              needs_update = true;
            }
          }
        }
      }

      if (needs_update) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        node_doc.Accept(writer);

        etcd_client_.set(node_key, buffer.GetString());
        global_logger->info("Updated node {} with new status and role", node_key);
      }
    }
  } catch (const std::exception &e) {
    global_logger->error("Exception while updating node states: {}", e.what());
  }

  curl_easy_cleanup(curl);

  // 采集各节点的 Raft 日志进度，供再平衡判断新节点是否已完成数据追平
  UpdateRaftProgress();
}

void MasterServiceImpl::UpdateRaftProgress() {
  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    global_logger->error("UpdateRaftProgress: CURL initialization failed");
    return;
  }

  try {
    // 1. 找出当前 leader，并建立 (instanceId, nodeId) -> etcd key 的映射。
    //    用组合键是因为不同实例下 nodeId 可能重复，单用 nodeId 会互相覆盖。
    EtcdResponse all = etcd_client_.ls("/instances/");
    std::string leader_url;
    std::map<std::pair<uint64_t, uint64_t>, std::string> node_keys;
    for (size_t i = 0; i < all.keys().size(); ++i) {
      rapidjson::Document node_doc;
      node_doc.Parse(all.values()[i].as_string().c_str());
      if (!node_doc.IsObject() || !node_doc.HasMember("status") || !node_doc.HasMember("role")) {
        continue;
      }
      uint64_t node_id = ParseTrailingId(all.keys()[i]);
      uint64_t instance_id = ParseNodeKeyInstanceId(all.keys()[i]);
      if (node_id == 0 || instance_id == 0) {
        continue;
      }
      node_keys[{instance_id, node_id}] = all.keys()[i];
      if (node_doc["status"].GetInt() == 1 && node_doc["role"].GetInt() == 0 && node_doc.HasMember("url")) {
        leader_url = node_doc["url"].GetString();
      }
    }

    if (leader_url.empty()) {
      global_logger->debug("UpdateRaftProgress: no alive leader, skip");
      curl_easy_cleanup(curl);
      return;
    }

    // 2. 向 leader 拉取全组各节点的日志进度
    std::string list_url = leader_url + "/AdminService/ListNode";
    curl_easy_setopt(curl, CURLOPT_URL, list_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, BaseServiceImpl::WriteCallback);
    std::string response_str;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      global_logger->warn("UpdateRaftProgress: ListNode request failed: {}", curl_easy_strerror(res));
      curl_easy_cleanup(curl);
      return;
    }

    rapidjson::Document doc;
    doc.Parse(response_str.c_str());
    if (!doc.IsObject() || !doc.HasMember("nodes") || !doc["nodes"].IsArray()) {
      global_logger->warn("UpdateRaftProgress: unexpected ListNode response");
      curl_easy_cleanup(curl);
      return;
    }

    // 3. 把各节点的日志进度写回 etcd（仅在变化时写入）
    for (const auto &node : doc["nodes"].GetArray()) {
      if (!node.HasMember("nodeId") || !node["nodeId"].IsInt() || !node.HasMember("last_log_idx") ||
          !node["last_log_idx"].IsUint64()) {
        continue;
      }
      uint64_t node_id = static_cast<uint64_t>(node["nodeId"].GetInt());
      uint64_t last_log_idx = node["last_log_idx"].GetUint64();

      // 同一 nodeId 可能属于多个实例；同处一个 Raft 组的日志进度一致，逐条写回
      for (const auto &entry : node_keys) {
        if (entry.first.second != node_id) {
          continue;
        }
        const std::string &node_key = entry.second;

        EtcdResponse resp = etcd_client_.get(node_key);
        if (!resp.is_ok()) {
          continue;
        }
        rapidjson::Document node_doc;
        node_doc.Parse(resp.value().as_string().c_str());
        if (!node_doc.IsObject()) {
          continue;
        }

        if (!node_doc.HasMember("lastLogIdx") || !node_doc["lastLogIdx"].IsUint64()) {
          node_doc.AddMember("lastLogIdx", rapidjson::Value(last_log_idx), node_doc.GetAllocator());
        } else if (node_doc["lastLogIdx"].GetUint64() == last_log_idx) {
          continue;  // 无变化，跳过写入
        } else {
          node_doc["lastLogIdx"].SetUint64(last_log_idx);
        }

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        node_doc.Accept(writer);
        etcd_client_.set(node_key, buffer.GetString());
      }
    }

    // 4. 补齐 Raft 成员：etcd 中已登记、状态正常、但尚未出现在 Raft 组内的节点，
    //    交由 leader 执行加入。这是"加机器只需启动进程"闭环的关键一步。
    std::set<uint64_t> raft_members;
    for (const auto &node : doc["nodes"].GetArray()) {
      if (node.HasMember("nodeId") && node["nodeId"].IsInt()) {
        raft_members.insert(static_cast<uint64_t>(node["nodeId"].GetInt()));
      }
    }

    for (const auto &entry : node_keys) {
      uint64_t node_id = entry.first.second;
      const std::string &node_key = entry.second;
      if (raft_members.find(node_id) != raft_members.end()) {
        continue;  // 已在组内
      }

      EtcdResponse resp = etcd_client_.get(node_key);
      if (!resp.is_ok()) {
        continue;
      }
      rapidjson::Document node_doc;
      node_doc.Parse(resp.value().as_string().c_str());
      if (!node_doc.IsObject()) {
        continue;
      }
      // 只把存活且已知 Raft 地址的节点拉入组内
      if (!node_doc.HasMember("status") || node_doc["status"].GetInt() != 1 ||
          !node_doc.HasMember("endpoint") || !node_doc["endpoint"].IsString()) {
        continue;
      }

      std::string endpoint = node_doc["endpoint"].GetString();
      if (JoinRaftGroup(leader_url, node_id, endpoint)) {
        global_logger->info("UpdateRaftProgress: node {} joined raft group", node_id);
      }
    }
  } catch (const std::exception &e) {
    global_logger->error("UpdateRaftProgress exception: {}", e.what());
  }

  curl_easy_cleanup(curl);
}

auto MasterServiceImpl::JoinRaftGroup(const std::string &leader_url, uint64_t node_id,
                                      const std::string &endpoint) -> bool {
  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    global_logger->error("JoinRaftGroup: CURL initialization failed");
    return false;
  }

  std::string url = leader_url + "/AdminService/AddFollower";

  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();
  doc.AddMember("nodeId", static_cast<int>(node_id), allocator);
  doc.AddMember("endpoint", rapidjson::Value(endpoint.c_str(), allocator), allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  std::string response_str;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.GetString());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, BaseServiceImpl::WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);
  // AddSrv 内部会轮询等待新成员生效，超时给得宽松一些
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

  CURLcode res = curl_easy_perform(curl);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    global_logger->warn("JoinRaftGroup: request to {} failed: {}", url, curl_easy_strerror(res));
    return false;
  }

  rapidjson::Document resp;
  resp.Parse(response_str.c_str());
  if (resp.HasParseError() || !resp.IsObject() || !resp.HasMember("retCode")) {
    global_logger->warn("JoinRaftGroup: unexpected response {}", response_str);
    return false;
  }
  if (resp["retCode"].GetInt() != 0) {
    global_logger->warn("JoinRaftGroup: leader rejected node {}: {}", node_id,
                        resp.HasMember("msg") && resp["msg"].IsString() ? resp["msg"].GetString() : "unknown");
    return false;
  }
  return true;
}

void MasterServiceImpl::GetPartitionConfig(::google::protobuf::RpcController *controller,
                                           const ::nvm::HttpRequest * /*request*/, ::nvm::HttpResponse * /*response*/,
                                           ::google::protobuf::Closure *done) {
  global_logger->info("Received GetPartitionConfig request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 从JSON请求中获取ID
  uint64_t instance_id = json_request[INSTANCE_ID].GetUint64();
  try {
    PartitionConfig config = DoGetPartitionConfig(instance_id);

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    // 添加分区配置信息到响应
    doc.AddMember("partitionKey", rapidjson::Value(config.partition_key_.c_str(), allocator), allocator);
    doc.AddMember("numberOfPartitions", config.number_of_partitions_, allocator);

    rapidjson::Value partitions_array(rapidjson::kArrayType);
    for (const auto &partition : config.partitions_) {
      rapidjson::Value partition_obj(rapidjson::kObjectType);
      partition_obj.AddMember("partitionId", partition.partition_id_, allocator);
      partition_obj.AddMember("nodeId", partition.node_id_, allocator);
      partitions_array.PushBack(partition_obj, allocator);
    }
    doc.AddMember("partitions", partitions_array, allocator);

    SetResponse(cntl, 0, "Partition config retrieved successfully", &doc);
  } catch (const std::exception &e) {
    SetResponse(cntl, 1, "Exception occurred: " + std::string(e.what()));
  }
}





void MasterServiceImpl::UpdatePartitionConfig(::google::protobuf::RpcController *controller,
                                              const ::nvm::HttpRequest * /*request*/,
                                              ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
 global_logger->info("Received AddNode request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  try {
    uint64_t instance_id = json_request["instanceId"].GetUint64();
    std::string partition_key = json_request["partitionKey"].GetString();
    int number_of_partitions = json_request["numberOfPartitions"].GetInt();

    std::list<Partition> partition_list;
    const rapidjson::Value &partitions = json_request["partitions"];
    for (const auto &partition : partitions.GetArray()) {
      uint64_t partition_id = partition["partitionId"].GetUint64();
      uint64_t node_id = partition["nodeId"].GetUint64();
      partition_list.push_back({partition_id, node_id});
    }
    DoUpdatePartitionConfig(instance_id, partition_key, number_of_partitions, partition_list);
    SetResponse(cntl, 0, "Partition configuration updated successfully");
  } catch (const std::exception &e) {
    SetResponse(cntl, 1, std::string("Error updating partition config: ") + e.what());
  }
}

void MasterServiceImpl::RunBalancer() {
  try {
    // 找出所有已配置分区表的实例
    EtcdResponse configs = etcd_client_.ls("/instancesConfig/");
    for (size_t i = 0; i < configs.keys().size(); ++i) {
      uint64_t instance_id = ParseConfigInstanceId(configs.keys()[i]);
      if (instance_id == 0) {
        continue;
      }
      RebalanceInstance(instance_id);
    }
  } catch (const std::exception &e) {
    global_logger->error("RunBalancer exception: {}", e.what());
  }
}

void MasterServiceImpl::RebalanceInstance(uint64_t instance_id) {
  // 1. 读取分区表
  PartitionConfig config = DoGetPartitionConfig(instance_id);
  if (config.partitions_.empty()) {
    return;  // 尚未划分分区，无从再平衡
  }

  // 2. 读取节点表，组装负载快照
  std::string nodes_prefix = "/instances/" + std::to_string(instance_id) + "/nodes/";
  EtcdResponse nodes = etcd_client_.ls(nodes_prefix);

  uint64_t leader_log_idx = 0;
  bool has_leader_progress = false;
  std::vector<NodeLoad> node_loads;
  std::map<uint64_t, uint64_t> node_log_idx;  // nodeId -> lastLogIdx

  for (size_t i = 0; i < nodes.keys().size(); ++i) {
    rapidjson::Document node_doc;
    node_doc.Parse(nodes.values()[i].as_string().c_str());
    if (!node_doc.IsObject() || !node_doc.HasMember("nodeId")) {
      continue;
    }

    NodeLoad load;
    load.node_id_ = node_doc["nodeId"].GetUint64();
    if (node_doc.HasMember("vectorCount") && node_doc["vectorCount"].IsUint64()) {
      load.vector_count_ = node_doc["vectorCount"].GetUint64();
    }
    if (node_doc.HasMember("diskUsageBytes") && node_doc["diskUsageBytes"].IsUint64()) {
      load.disk_usage_bytes_ = node_doc["diskUsageBytes"].GetUint64();
    }

    bool alive = node_doc.HasMember("status") && node_doc["status"].GetInt() == 1;
    bool is_leader = node_doc.HasMember("role") && node_doc["role"].GetInt() == 0;

    if (node_doc.HasMember("lastLogIdx") && node_doc["lastLogIdx"].IsUint64()) {
      uint64_t idx = node_doc["lastLogIdx"].GetUint64();
      node_log_idx[load.node_id_] = idx;
      if (is_leader) {
        leader_log_idx = idx;
        has_leader_progress = true;
      }
    }

    // 先按存活判定，ready 需结合日志进度在下面统一计算
    load.ready_ = alive;
    node_loads.push_back(load);
  }

  // 3. 结合 leader 的日志进度判定各节点是否已完成追平。
  //    拿不到 leader 进度时保守地认为全部未就绪，避免在信息不足时盲目调度。
  if (!has_leader_progress) {
    global_logger->debug("Rebalance: instance {} has no leader progress, skip", instance_id);
    return;
  }
  for (auto &load : node_loads) {
    auto it = node_log_idx.find(load.node_id_);
    if (it == node_log_idx.end()) {
      // 从未上报过日志进度：通常是刚加入、尚未进入 Raft 组，不可接管分区
      load.ready_ = false;
      continue;
    }
    load.ready_ = load.ready_ && HasCaughtUp(it->second, leader_log_idx);
  }

  // 4. 决策（纯函数）
  std::vector<std::pair<uint64_t, uint64_t>> partition_to_node;
  partition_to_node.reserve(config.partitions_.size());
  for (const auto &partition : config.partitions_) {
    partition_to_node.emplace_back(partition.partition_id_, partition.node_id_);
  }

  auto plan = PlanMigration(partition_to_node, node_loads);
  if (!plan.has_value()) {
    return;  // 已均衡或无可迁移目标
  }

  // 5. 单飞：同一分区同时只允许一个迁移任务
  std::string migration_key = std::to_string(instance_id) + ":" + std::to_string(plan->partition_id_);
  {
    std::lock_guard<std::mutex> lock(migration_mutex_);
    if (active_migrations_.find(migration_key) != active_migrations_.end()) {
      global_logger->debug("Rebalance: migration {} still in progress, skip", migration_key);
      return;
    }
    active_migrations_.insert(migration_key);
  }

  // 6. 执行：改写分区归属并写回 etcd，Proxy 会在下个周期拉到新拓扑
  bool changed = false;
  for (auto &partition : config.partitions_) {
    if (partition.partition_id_ == plan->partition_id_) {
      if (partition.node_id_ != plan->to_node_id_) {
        global_logger->info("Rebalance: instance {} partition {} : node {} -> node {}", instance_id,
                            plan->partition_id_, partition.node_id_, plan->to_node_id_);
        partition.node_id_ = plan->to_node_id_;
        changed = true;
      }
      break;
    }
  }

  if (changed) {
    DoUpdatePartitionConfig(instance_id, config.partition_key_, config.number_of_partitions_, config.partitions_);
    global_logger->info("Rebalance: instance {} partition config updated", instance_id);
  }

  {
    std::lock_guard<std::mutex> lock(migration_mutex_);
    active_migrations_.erase(migration_key);
  }
}

auto MasterServiceImpl::DoGetPartitionConfig(uint64_t instanceId) -> PartitionConfig {
    PartitionConfig config;
    std::string etcd_key = "/instancesConfig/" + std::to_string(instanceId) + "/partitionConfig";
    EtcdResponse etcd_response = etcd_client_.get(etcd_key);
    rapidjson::Document doc;
    doc.Parse(etcd_response.value().as_string().c_str());

    if (doc.IsObject()) {
        if (doc.HasMember("partitionKey")) {
            config.partition_key_ = doc["partitionKey"].GetString();
        }
        if (doc.HasMember("numberOfPartitions")) {
            config.number_of_partitions_ = doc["numberOfPartitions"].GetInt();
        }
        if (doc.HasMember("partitions") && doc["partitions"].IsArray()) {
            for (const auto& partition : doc["partitions"].GetArray()) {
                Partition p;
                p.partition_id_ = partition["partitionId"].GetUint64();
                p.node_id_ = partition["nodeId"].GetUint64();
                config.partitions_.push_back(p);
            }
        }
    }

    return config;
}

void MasterServiceImpl::DoUpdatePartitionConfig(uint64_t instanceId, const std::string& partitionKey, int numberOfPartitions, const std::list<Partition>& partitions) {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

    // 设置分区键和分区数目
    doc.AddMember("partitionKey", rapidjson::Value(partitionKey.c_str(), allocator), allocator);
    doc.AddMember("numberOfPartitions", numberOfPartitions, allocator);

    // 设置分区映射
    rapidjson::Value partition_array(rapidjson::kArrayType);
    for (const auto& partition : partitions) {
        rapidjson::Value partition_obj(rapidjson::kObjectType);
        partition_obj.AddMember("partitionId", partition.partition_id_, allocator);
        partition_obj.AddMember("nodeId", partition.node_id_, allocator);
        partition_array.PushBack(partition_obj, allocator);
    }
    doc.AddMember("partitions", partition_array, allocator);

    // 将配置写入 etcd
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string etcd_key = "/instancesConfig/" + std::to_string(instanceId) + "/partitionConfig";
    etcd_client_.set(etcd_key, buffer.GetString());
    global_logger->info("Updated partition config for instance {}", instanceId);
}

}  // namespace vectordb