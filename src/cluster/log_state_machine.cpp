#include "cluster/log_state_machine.h"
#include <iostream>
#include "common/constants.h"
#include "logger/logger.h"

namespace vectordb {

namespace {
// 从 JSON 请求中解析 collectionName（默认 "default"）
auto GetCollectionNameFromRequest(const rapidjson::Document& json_request) -> std::string {
    if (json_request.HasMember(REQUEST_COLLECTION_NAME) && json_request[REQUEST_COLLECTION_NAME].IsString()) {
        return json_request[REQUEST_COLLECTION_NAME].GetString();
    }
    return DEFAULT_COLLECTION_NAME;
}
} // anonymous namespace

void LogStateMachine::SetVectorDatabase(VectorDatabase *vector_database) {
  vector_database_ = vector_database;  // 设置 vector_database_ 指针
  last_committed_idx_ = vector_database->GetStartIndexId();
}

auto LogStateMachine::commit(const nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> {
  std::string content(reinterpret_cast<const char *>(data.data() + data.pos() + sizeof(int)),
                      data.size() - sizeof(int));
  global_logger->debug("Commit log_idx: {}, content: {}", log_idx, content);

  rapidjson::Document json_request;
  json_request.Parse(content.c_str());

  // Update last committed index number.
  last_committed_idx_ = log_idx;

  // 解析 collectionName
  std::string collection_name = GetCollectionNameFromRequest(json_request);

  // 检查是否为批量插入操作
  if (json_request.HasMember(REQUEST_OPERATION_TYPE) &&
      json_request[REQUEST_OPERATION_TYPE].IsString() &&
      std::string(json_request[REQUEST_OPERATION_TYPE].GetString()) == OPERATION_TYPE_BATCH_UPSERT) {
    // 批量插入操作
    IndexFactory::IndexType index_type = vector_database_->GetIndexTypeFromRequest(json_request);
    if (json_request.HasMember(REQUEST_ITEMS) && json_request[REQUEST_ITEMS].IsArray()) {
      const auto& items = json_request[REQUEST_ITEMS].GetArray();
      std::vector<uint64_t> ids;
      std::vector<rapidjson::Document> datas;
      ids.reserve(items.Size());
      datas.reserve(items.Size());
      for (rapidjson::SizeType i = 0; i < items.Size(); ++i) {
        const auto& item = items[i];
        if (!item.IsObject() || !item.HasMember(REQUEST_ID) || !item.HasMember(REQUEST_VECTORS)) {
          continue;
        }
        uint64_t id = item[REQUEST_ID].GetUint64();
        rapidjson::Document doc;
        doc.CopyFrom(item, doc.GetAllocator());
        ids.push_back(id);
        datas.push_back(std::move(doc));
      }
      if (!ids.empty()) {
        vector_database_->BatchUpsert(collection_name, ids, datas, index_type);
      }
    }
  } else {
    // 单条插入操作
    uint64_t label = json_request[REQUEST_ID].GetUint64();
    IndexFactory::IndexType index_type = vector_database_->GetIndexTypeFromRequest(json_request);
    vector_database_->Upsert(collection_name, label, json_request, index_type);
  }

  // Return Raft log number as a return result.
  nuraft::ptr<nuraft::buffer> ret = nuraft::buffer::alloc(sizeof(log_idx));
  nuraft::buffer_serializer bs(ret);
  bs.put_u64(log_idx);
  return ret;
}

auto LogStateMachine::pre_commit(const nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> {
  std::string content(reinterpret_cast<const char *>(data.data() + data.pos() + sizeof(int)),
                      data.size() - sizeof(int));
  global_logger->debug("Pre Commit log_idx: {}, content: {}", log_idx, content);  // 添加打印日志
  return nullptr;
}

}  // namespace vectordb