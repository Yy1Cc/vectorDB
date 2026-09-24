#include "httpserver/admin_service_impl.h"
#include <brpc/controller.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include "common/constants.h"
#include "common/vector_cfg.h"
#include "collection/collection_manager.h"
#include "index/faiss_index.h"
#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "logger/logger.h"

namespace vectordb {

namespace {
// 统计路径占用的字节数：目录则递归累加其中所有常规文件，单文件则取其大小。
// 路径不存在或无访问权限时返回 0，不抛异常（负载采集失败不应影响节点状态上报）。
auto ComputePathSize(const std::string &path) -> uint64_t {
  if (path.empty()) {
    return 0;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec) || ec) {
    return 0;
  }
  if (std::filesystem::is_regular_file(path, ec) && !ec) {
    auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
  }
  if (!std::filesystem::is_directory(path, ec) || ec) {
    return 0;
  }

  uint64_t total = 0;
  for (std::filesystem::recursive_directory_iterator it(path, ec), end; it != end; it.increment(ec)) {
    if (ec) {  // 无权限或目录在遍历中被删除，停止累计，返回已统计部分
      break;
    }
    if (it->is_regular_file(ec) && !ec) {
      auto size = it->file_size(ec);
      if (!ec) {
        total += static_cast<uint64_t>(size);
      }
    }
  }
  return total;
}
}  // namespace
void AdminServiceImpl::snapshot(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                                ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received snapshot request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  vector_database_->TakeSnapshot();  // 调用 VectorDatabase::takeSnapshot

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::SetLeader(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                                 ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received setLeader request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  // 将当前节点设置为主节点
  raft_stuff_->EnableElectionTimeout(10000, 20000);

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::AddFollower(::google::protobuf::RpcController *controller,
                                   const ::nvm::HttpRequest * /*request*/, ::nvm::HttpResponse * /*response*/,
                                   ::google::protobuf::Closure *done) {
  global_logger->debug("Received addFollower request");

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

  // 检查当前节点是否为leader
  if (!raft_stuff_->IsLeader()) {
    global_logger->error("Current node is not the leader");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Current node is not the leader");
    return;
  }

  // 从JSON请求中获取follower节点信息
  int node_id = json_request["nodeId"].GetInt();
  std::string endpoint = json_request["endpoint"].GetString();

  // 默认以 learner 身份加入：不计入法定人数，后台追赶，不影响线上写入。
  // 传 "learner": false 才走旧行为（立即计入法定人数，追赶期间写入会阻塞）。
  bool as_learner = true;
  if (json_request.HasMember("learner") && json_request["learner"].IsBool()) {
    as_learner = json_request["learner"].GetBool();
  }

  // 调用 RaftStuff 的 addSrv 方法将新的follower节点添加到集群中
  bool success = raft_stuff_->AddSrv(node_id, endpoint, as_learner);

  if (!success) {
    global_logger->error("raft_stuff  AddSrv  failed");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "raft_stuff  AddSrv  failed");
    return;
  }

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  json_response.AddMember("learner", as_learner, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::PromoteLearner(::google::protobuf::RpcController *controller,
                                       const ::nvm::HttpRequest * /*request*/,
                                       ::nvm::HttpResponse * /*response*/,
                                       ::google::protobuf::Closure *done) {
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  if (!json_request.IsObject() || !json_request.HasMember("nodeId")) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing nodeId");
    return;
  }

  if (!raft_stuff_->IsLeader()) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Current node is not the leader");
    return;
  }

  int node_id = json_request["nodeId"].GetInt();

  // 只有追平后才转正，否则法定人数提高后新写入会被未追上的节点卡住。
  const auto peer_idx = raft_stuff_->GetPeerLastLogIdx(node_id);
  const auto committed_idx = raft_stuff_->GetCommittedLogIdx();
  if (peer_idx + 1 < committed_idx) {
    global_logger->warn("PromoteLearner rejected: srv {} last_log_idx={} < committed {}",
                        node_id, peer_idx, committed_idx);
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR,
                         "Learner has not caught up yet (lastLogIdx behind committedIdx)");
    return;
  }

  if (!raft_stuff_->PromoteLearner(node_id)) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Failed to promote learner");
    return;
  }

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  json_response.AddMember("lastLogIdx", static_cast<uint64_t>(peer_idx), allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::RemoveFollower(::google::protobuf::RpcController *controller,
                                      const ::nvm::HttpRequest * /*request*/, ::nvm::HttpResponse * /*response*/,
                                      ::google::protobuf::Closure *done) {
  global_logger->debug("Received removeFollower request");

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

  if (!json_request.HasMember("nodeId") || !json_request["nodeId"].IsInt()) {
    global_logger->error("Missing or invalid 'nodeId'");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing or invalid 'nodeId'");
    return;
  }

  // 检查当前节点是否为leader
  if (!raft_stuff_->IsLeader()) {
    global_logger->error("Current node is not the leader");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Current node is not the leader");
    return;
  }

  int node_id = json_request["nodeId"].GetInt();

  // 调用 RaftStuff 的 RemoveSrv 方法将 follower 节点从集群中摘除
  bool success = raft_stuff_->RemoveSrv(node_id);

  if (!success) {
    global_logger->error("raft_stuff RemoveSrv failed for node {}", node_id);
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "raft_stuff RemoveSrv failed");
    return;
  }

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::ListNode(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                                ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received listNode request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  // 获取所有节点信息
  auto nodes_info = raft_stuff_->GetAllNodesInfo();

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 将节点信息添加到JSON响应中
  rapidjson::Value nodes_array(rapidjson::kArrayType);
  for (const auto &node_info : nodes_info) {
    rapidjson::Value node_object(rapidjson::kObjectType);
    node_object.AddMember("nodeId", node_info.node_id, allocator);
    node_object.AddMember("endpoint", rapidjson::Value(node_info.endpoint.c_str(), allocator), allocator);
    node_object.AddMember("state", rapidjson::Value(node_info.role.c_str(), allocator), allocator);
    // learner 不计入法定人数、也不发起选举。Master 据此判定谁需要自动转正。
    node_object.AddMember("learner", node_info.is_learner, allocator);
    node_object.AddMember("last_log_idx", node_info.last_log_idx, allocator);
    node_object.AddMember("last_succ_resp_us", node_info.last_succ_resp_us, allocator);
    nodes_array.PushBack(node_object, allocator);
  }
  json_response.AddMember("nodes", nodes_array, allocator);

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::GetNode(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                               ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received getNode request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  // 获取所有节点信息
  std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong> node_info =
      raft_stuff_->GetCurrentNodesInfo();

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 将节点信息添加到JSON响应中
  rapidjson::Value nodes_array(rapidjson::kArrayType);
  rapidjson::Value node_object(rapidjson::kObjectType);
  node_object.AddMember("nodeId", std::get<0>(node_info), allocator);
  node_object.AddMember("endpoint", rapidjson::Value(std::get<1>(node_info).c_str(), allocator), allocator);
  node_object.AddMember("state", rapidjson::Value(std::get<2>(node_info).c_str(), allocator),
                        allocator);                                               // 添加节点状态
  node_object.AddMember("last_log_idx", std::get<3>(node_info), allocator);       // 添加节点最后日志索引
  node_object.AddMember("last_succ_resp_us", std::get<4>(node_info), allocator);  // 添加节点最后成功响应时间

  // 负载指标：向量总条数与磁盘占用，供 Master 采集后用于再平衡决策
  uint64_t vector_count = 0;
  for (const auto &name : CollectionManager::Instance().ListCollections()) {
    auto *coll = CollectionManager::Instance().GetCollection(name);
    if (coll != nullptr) {
      vector_count += static_cast<uint64_t>(coll->index_factory.GetTotalCount());
    }
  }
  uint64_t disk_usage_bytes = ComputePathSize(Cfg::Instance().RocksDbPath()) +
                              ComputePathSize(Cfg::Instance().WalPath()) +
                              ComputePathSize(Cfg::Instance().SnapPath());

  node_object.AddMember("vectorCount", vector_count, allocator);
  node_object.AddMember("diskUsageBytes", disk_usage_bytes, allocator);

  json_response.AddMember("node", node_object, allocator);

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::createCollection(::google::protobuf::RpcController *controller,
                                         const ::nvm::HttpRequest * /*request*/,
                                         ::nvm::HttpResponse * /*response*/,
                                         ::google::protobuf::Closure *done) {
  global_logger->debug("Received createCollection request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  if (!json_request.IsObject()) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  if (!json_request.HasMember("name") || !json_request.HasMember("dimension")) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing name or dimension");
    return;
  }

  std::string name = json_request["name"].GetString();
  int dim = json_request["dimension"].GetInt();
  int num_data = json_request.HasMember("numData") ? json_request["numData"].GetInt() : 1000000;
  IndexFactory::MetricType metric = IndexFactory::MetricType::L2;
  if (json_request.HasMember("metric") && std::string(json_request["metric"].GetString()) == "IP") {
    metric = IndexFactory::MetricType::IP;
  }

  // 建集合属于元数据变更，必须经 Raft 复制到所有副本。
  // 此前只在接收请求的节点本地创建：其余副本上该 collection 不存在，
  // 重放 upsert 时会用各自的向量维度自动创建，GARDEN 子图结构也随之不一致。
  rapidjson::Document repl;
  repl.SetObject();
  auto &repl_alloc = repl.GetAllocator();
  repl.AddMember(REQUEST_OPERATION_TYPE, rapidjson::Value(OPERATION_TYPE_CREATE_COLLECTION, repl_alloc), repl_alloc);
  repl.AddMember(REQUEST_COLLECTION_NAME, rapidjson::Value(name.c_str(), repl_alloc), repl_alloc);
  repl.AddMember("dim", dim, repl_alloc);
  repl.AddMember("numData", num_data, repl_alloc);
  repl.AddMember("metric", rapidjson::Value(metric == IndexFactory::MetricType::IP ? "IP" : "L2", repl_alloc),
                 repl_alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  repl.Accept(writer);

  bool success = raft_stuff_->AppendEntries(buffer.GetString());

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();
  json_response.AddMember(RESPONSE_RETCODE, success ? RESPONSE_RETCODE_SUCCESS : RESPONSE_RETCODE_ERROR, allocator);
  if (!success) {
    json_response.AddMember(RESPONSE_ERROR_MSG,
                            rapidjson::Value("Failed to replicate createCollection via Raft", allocator), allocator);
  }
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::dropCollection(::google::protobuf::RpcController *controller,
                                       const ::nvm::HttpRequest * /*request*/,
                                       ::nvm::HttpResponse * /*response*/,
                                       ::google::protobuf::Closure *done) {
  global_logger->debug("Received dropCollection request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  if (!json_request.IsObject() || !json_request.HasMember("name")) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing name");
    return;
  }

  std::string name = json_request["name"].GetString();
  bool success = CollectionManager::Instance().DropCollection(name);

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();
  json_response.AddMember(RESPONSE_RETCODE, success ? RESPONSE_RETCODE_SUCCESS : RESPONSE_RETCODE_ERROR, allocator);
  if (!success) {
    json_response.AddMember(RESPONSE_ERROR_MSG, rapidjson::Value("Collection not found or is default", allocator), allocator);
  }
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::listCollections(::google::protobuf::RpcController *controller,
                                        const ::nvm::HttpRequest * /*request*/,
                                        ::nvm::HttpResponse * /*response*/,
                                        ::google::protobuf::Closure *done) {
  global_logger->debug("Received listCollections request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  auto names = CollectionManager::Instance().ListCollections();

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  rapidjson::Value collections_array(rapidjson::kArrayType);
  for (const auto &name : names) {
    collections_array.PushBack(rapidjson::Value(name.c_str(), allocator), allocator);
  }
  json_response.AddMember("collections", collections_array, allocator);
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::getCollectionInfo(::google::protobuf::RpcController *controller,
                                          const ::nvm::HttpRequest * /*request*/,
                                          ::nvm::HttpResponse * /*response*/,
                                          ::google::protobuf::Closure *done) {
  global_logger->debug("Received getCollectionInfo request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  if (!json_request.IsObject() || !json_request.HasMember("name")) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing name");
    return;
  }

  std::string name = json_request["name"].GetString();
  CollectionMeta meta;
  bool found = CollectionManager::Instance().GetCollectionMeta(name, meta);

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  if (found) {
    json_response.AddMember("name", rapidjson::Value(meta.name.c_str(), allocator), allocator);
    json_response.AddMember("dimension", meta.dimension, allocator);
    json_response.AddMember("numData", meta.num_data, allocator);
    json_response.AddMember("metric", rapidjson::Value(meta.metric == IndexFactory::MetricType::IP ? "IP" : "L2", allocator), allocator);
    json_response.AddMember("createdAt", rapidjson::Value(meta.created_at.c_str(), allocator), allocator);
    json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  } else {
    json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_ERROR, allocator);
    json_response.AddMember(RESPONSE_ERROR_MSG, rapidjson::Value("Collection not found", allocator), allocator);
  }
  SetJsonResponse(json_response, cntl);
}

void AdminServiceImpl::registerGardenField(::google::protobuf::RpcController *controller,
                                           const ::nvm::HttpRequest * /*request*/,
                                           ::nvm::HttpResponse * /*response*/,
                                           ::google::protobuf::Closure *done) {
  global_logger->debug("Received registerGardenField request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  if (!json_request.IsObject()) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 必填字段：collectionName / field / fieldType
  if (!json_request.HasMember("collectionName") || !json_request.HasMember("field") ||
      !json_request.HasMember("fieldType")) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing collectionName / field / fieldType");
    return;
  }

  std::string collection_name = json_request["collectionName"].GetString();
  std::string field = json_request["field"].GetString();
  std::string field_type = json_request["fieldType"].GetString();

  auto *coll = CollectionManager::Instance().GetCollection(collection_name);
  if (coll == nullptr) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Collection not found");
    return;
  }

  auto *garden = static_cast<GardenIndex *>(coll->GetIndex(IndexFactory::IndexType::GARDEN_HNSW));
  if (garden == nullptr) {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "GARDEN_HNSW index not initialized");
    return;
  }

  // 字段注册属于元数据变更，必须经 Raft 复制到所有副本。
  // 此前只改本地内存：副本上没有连续字段的分桶子图（离散字段靠 upsert 时的
  // 懒注册侥幸收敛，连续字段必须显式 min/max/bucket 而无法自动注册），
  // 导致同一查询在 leader 走子图、在 follower 走全量图，副本间延迟差 2-4 倍。
  rapidjson::Document repl;
  repl.SetObject();
  auto &repl_alloc = repl.GetAllocator();
  repl.AddMember(REQUEST_OPERATION_TYPE,
                 rapidjson::Value(OPERATION_TYPE_REGISTER_GARDEN_FIELD, repl_alloc), repl_alloc);
  repl.AddMember(REQUEST_COLLECTION_NAME, rapidjson::Value(collection_name.c_str(), repl_alloc), repl_alloc);
  repl.AddMember("field", rapidjson::Value(field.c_str(), repl_alloc), repl_alloc);
  repl.AddMember("fieldType", rapidjson::Value(field_type.c_str(), repl_alloc), repl_alloc);

  if (field_type == "discrete") {
    // 离散字段：仅注册字段名，子图在 upsert 时按值懒创建
    global_logger->info("registerGardenField: discrete field '{}' on collection '{}'", field, collection_name);
  } else if (field_type == "continuous") {
    // 连续字段：必须提供 min / max，bucketSize 可选（默认 kBruteBound）
    if (!json_request.HasMember("min") || !json_request.HasMember("max")) {
      cntl->http_response().set_status_code(400);
      SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Continuous field requires min and max");
      return;
    }
    int64_t min_val = json_request["min"].GetInt64();
    int64_t max_val = json_request["max"].GetInt64();
    int bucket_size = GardenIndex::kBruteBound;
    if (json_request.HasMember("bucketSize") && json_request["bucketSize"].IsInt()) {
      bucket_size = json_request["bucketSize"].GetInt();
    }

    if (min_val >= max_val || bucket_size <= 0) {
      cntl->http_response().set_status_code(400);
      SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid range or bucketSize");
      return;
    }
    repl.AddMember("min", min_val, repl_alloc);
    repl.AddMember("max", max_val, repl_alloc);
    repl.AddMember("bucketSize", bucket_size, repl_alloc);
    global_logger->info("registerGardenField: continuous field '{}' on '{}' range=[{},{}] bucket={}",
                        field, collection_name, min_val, max_val, bucket_size);
  } else {
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "fieldType must be 'discrete' or 'continuous'");
    return;
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  repl.Accept(writer);

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  if (!raft_stuff_->AppendEntries(buffer.GetString())) {
    global_logger->error("registerGardenField failed: Raft replication did not succeed");
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Failed to replicate registerGardenField via Raft");
    return;
  }

  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  json_response.AddMember("fieldType", rapidjson::Value(field_type.c_str(), allocator), allocator);
  SetJsonResponse(json_response, cntl);
}

}  // namespace vectordb