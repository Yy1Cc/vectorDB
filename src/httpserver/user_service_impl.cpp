#include "httpserver/user_service_impl.h"
#include <brpc/controller.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include "common/constants.h"
#include "collection/collection_manager.h"
#include "index/faiss_index.h"
#include "index/hnswlib_index.h"
#include "index/layered_index.h"
#include "index/index_factory.h"
#include "logger/logger.h"
namespace vectordb {
void UserServiceImpl::search(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                             ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received search request");

  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  global_logger->info("Search request parameters: {}", cntl->request_attachment().to_string());

  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    done->Run();
    return;
  }

  if (!IsRequestValid(json_request, BaseServiceImpl::CheckType::SEARCH)) {
    global_logger->error("Missing vectors or k parameter in the request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing vectors or k parameter in the request");
    done->Run();
    return;
  }

  std::vector<float> query;
  for (const auto &q : json_request[REQUEST_VECTORS].GetArray()) {
    query.push_back(q.GetFloat());
  }
  int k = json_request[REQUEST_K].GetInt();

  global_logger->debug("Query parameters: k = {}", k);

  IndexFactory::IndexType index_type = GetIndexTypeFromRequest(json_request);

  if (index_type == IndexFactory::IndexType::UNKNOWN) {
    global_logger->error("Invalid indexType parameter in the request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid indexType parameter in the request");
    done->Run();
    return;
  }

  // 解析 collectionName（默认 "default"）
  std::string collection_name = DEFAULT_COLLECTION_NAME;
  if (json_request.HasMember(REQUEST_COLLECTION_NAME) && json_request[REQUEST_COLLECTION_NAME].IsString()) {
    collection_name = json_request[REQUEST_COLLECTION_NAME].GetString();
  }

  std::pair<std::vector<int64_t>, std::vector<float>> results = vector_database_->Search(collection_name, json_request);

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  bool valid_results = false;
  rapidjson::Value vectors(rapidjson::kArrayType);
  rapidjson::Value distances(rapidjson::kArrayType);
  for (size_t i = 0; i < results.first.size(); ++i) {
    if (results.first[i] != -1) {
      valid_results = true;
      vectors.PushBack(results.first[i], allocator);
      distances.PushBack(results.second[i], allocator);
    }
  }

  if (valid_results) {
    json_response.AddMember(RESPONSE_VECTORS, vectors, allocator);
    json_response.AddMember(RESPONSE_DISTANCES, distances, allocator);
  }

  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

void UserServiceImpl::insert(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                             ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received insert request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  global_logger->info("Insert request parameters: {}", cntl->request_attachment().to_string());

  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  if (!IsRequestValid(json_request, BaseServiceImpl::CheckType::INSERT)) {
    global_logger->error("Missing vectors or id parameter in the request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing vectors or k parameter in the request");
    return;
  }

  // 校验 vectors 是合法的浮点数组（提前拦下格式错误，避免把坏日志写进 Raft）
  if (!json_request[REQUEST_VECTORS].IsArray()) {
    global_logger->error("vectors must be an array");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "vectors must be an array");
    return;
  }
  for (const auto &d : json_request[REQUEST_VECTORS].GetArray()) {
    if (!d.IsNumber()) {
      global_logger->error("Invalid element in vectors array");
      cntl->http_response().set_status_code(400);
      SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "vectors must be an array of numbers");
      return;
    }
  }

  IndexFactory::IndexType index_type = GetIndexTypeFromRequest(json_request);

  if (index_type == IndexFactory::IndexType::UNKNOWN) {
    global_logger->error("Invalid indexType parameter in the request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid indexType parameter in the request");
    return;
  }

  // insert 必须经 Raft 复制：此前它直接写本地索引，既不会同步到其它副本，
  // 也不写 WAL，数据只存在于接收到请求的那个节点上（proxy 轮询时随机命中）。
  // 重放侧走 VectorDatabase::Upsert（先删后插的幂等覆盖写），
  // 且 Upsert 已覆盖 GARDEN_HNSW —— 此前本地 switch 缺少该 case，
  // 对 GARDEN 索引调用 insert 会静默什么都不做却返回成功。
  if (!raft_stuff_->AppendEntries(cntl->request_attachment().to_string())) {
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Failed to replicate write via Raft");
    return;
  }

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);

  SetJsonResponse(json_response, cntl);
}

void UserServiceImpl::upsert(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                             ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received upsert request");
  brpc::ClosureGuard done_guard(done);
  auto *cntl = static_cast<brpc::Controller *>(controller);

  // 解析JSON请求
  rapidjson::Document json_request;
  json_request.Parse(cntl->request_attachment().to_string().c_str());

  global_logger->info("Upsert request parameters: {}", cntl->request_attachment().to_string());

  // 检查JSON文档是否为有效对象
  if (!json_request.IsObject()) {
    global_logger->error("Invalid JSON request");
    cntl->http_response().set_status_code(400);
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Invalid JSON request");
    return;
  }

  // 检查请求的合法性
  if (!IsRequestValid(json_request, BaseServiceImpl::CheckType::UPSERT)) {
    // 检查是否为批量插入请求
    if (json_request.HasMember(REQUEST_OPERATION_TYPE) &&
        json_request[REQUEST_OPERATION_TYPE].IsString() &&
        std::string(json_request[REQUEST_OPERATION_TYPE].GetString()) == OPERATION_TYPE_BATCH_UPSERT &&
        json_request.HasMember(REQUEST_ITEMS) && json_request[REQUEST_ITEMS].IsArray()) {
      // 批量插入请求，跳过单条校验
      global_logger->info("Batch upsert request with {} items", json_request[REQUEST_ITEMS].Size());
    } else {
      global_logger->error("Missing vectors or id parameter in the request");
      cntl->http_response().set_status_code(400);
      SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Missing vectors or id parameter in the request");
      return;
    }
  }

  // 调用 RaftStuff 的 appendEntries 方法将新的日志条目添加到集群中。
  // 必须检查返回值：非 leader / 被拒绝 / 提交超时都会返回 false。
  // 此前无条件返回 retCode=0，导致写入失败时客户端以为成功、数据静默丢失。
  if (!raft_stuff_->AppendEntries(cntl->request_attachment().to_string())) {
    global_logger->error("Upsert failed: Raft replication did not succeed");
    SetErrorJsonResponse(cntl, RESPONSE_RETCODE_ERROR, "Failed to replicate write via Raft");
    return;
  }

  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &response_allocator = json_response.GetAllocator();

  // 添加retCode到响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, response_allocator);

  SetJsonResponse(json_response, cntl);
}

void UserServiceImpl::query(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                            ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) {
  global_logger->debug("Received query request");

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
  uint64_t id = json_request[REQUEST_ID].GetUint64();

  // 解析 collectionName（默认 "default"）
  std::string collection_name = DEFAULT_COLLECTION_NAME;
  if (json_request.HasMember(REQUEST_COLLECTION_NAME) && json_request[REQUEST_COLLECTION_NAME].IsString()) {
    collection_name = json_request[REQUEST_COLLECTION_NAME].GetString();
  }

  // 查询JSON数据
  rapidjson::Document json_data = vector_database_->Query(collection_name, id);

  // 将结果转换为JSON
  rapidjson::Document json_response;
  json_response.SetObject();
  rapidjson::Document::AllocatorType &allocator = json_response.GetAllocator();

  // 如果查询到向量，则将json_data对象的内容合并到json_response对象中
  if (!json_data.IsNull()) {
    for (auto it = json_data.MemberBegin(); it != json_data.MemberEnd(); ++it) {
      json_response.AddMember(it->name, it->value, allocator);
    }
  }

  // 设置响应
  json_response.AddMember(RESPONSE_RETCODE, RESPONSE_RETCODE_SUCCESS, allocator);
  SetJsonResponse(json_response, cntl);
}

}  // namespace vectordb