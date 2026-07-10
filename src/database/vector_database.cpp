#include "database/vector_database.h"
#include <rapidjson/document.h>
#include <cstdint>
#include <vector>
#include "common/constants.h"
#include "database/scalar_storage.h"
#include "index/faiss_index.h"
#include "index/filter_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "index/layered_index.h"
#include "logger/logger.h"
#include <rapidjson/stringbuffer.h> // 包含 rapidjson/stringbuffer.h 以使用 StringBuffer 类
#include <rapidjson/writer.h> // 包含 rapidjson/writer.h 以使用 Writer 类

namespace vectordb {

namespace {
// 解析过滤操作符字符串
FilterIndex::Operation ParseFilterOp(const std::string& op_str) {
    if (op_str == "=") return FilterIndex::Operation::EQUAL;
    if (op_str == "!=") return FilterIndex::Operation::NOT_EQUAL;
    if (op_str == ">") return FilterIndex::Operation::GREATER_THAN;
    if (op_str == "<") return FilterIndex::Operation::LESS_THAN;
    if (op_str == ">=") return FilterIndex::Operation::GREATER_EQUAL;
    if (op_str == "<=") return FilterIndex::Operation::LESS_EQUAL;
    return FilterIndex::Operation::EQUAL;
}
} // anonymous namespace

VectorDatabase::VectorDatabase(const std::string &db_path, const std::string& wal_path) : scalar_storage_(db_path) {
    persistence_.Init(wal_path); // 初始化 persistence_ 对象
}

void VectorDatabase::ReloadDatabase() {
    global_logger->info("Entering VectorDatabase::reloadDatabase()"); // 在方法开始时打印日志

    persistence_.LoadSnapshot();

    std::string operation_type;
    rapidjson::Document json_data;
    persistence_.ReadNextWalLog(&operation_type, &json_data); // 通过指针的方式调用 readNextWALLog

    while (!operation_type.empty()) {
        global_logger->info("Operation Type: {}", operation_type);

        // 打印读取的一行内容fmt::detail::buffer
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        json_data.Accept(writer);
        global_logger->info("Read Line: {}", buffer.GetString());

       if (operation_type == "upsert") {
            // 检查是否为批量操作
            if (json_data.HasMember(REQUEST_OPERATION_TYPE) &&
                json_data[REQUEST_OPERATION_TYPE].IsString() &&
                std::string(json_data[REQUEST_OPERATION_TYPE].GetString()) == OPERATION_TYPE_BATCH_UPSERT) {
                // 批量插入
                IndexFactory::IndexType index_type = GetIndexTypeFromRequest(json_data);
                if (json_data.HasMember(REQUEST_ITEMS) && json_data[REQUEST_ITEMS].IsArray()) {
                    const auto& items = json_data[REQUEST_ITEMS].GetArray();
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
                        BatchUpsert(ids, datas, index_type);
                    }
                }
            } else {
                uint64_t id = json_data[REQUEST_ID].GetUint64();
                IndexFactory::IndexType index_type = GetIndexTypeFromRequest(json_data);
                Upsert(id, json_data, index_type); // 调用 VectorDatabase::upsert 接口重建数据
            }
        }

        // 清空 json_data
        rapidjson::Document().Swap(json_data);

        // 读取下一条 WAL 日志
        operation_type.clear();
        persistence_.ReadNextWalLog(&operation_type, &json_data);
    }
}

void VectorDatabase::WriteWalLog(const std::string& operation_type, const rapidjson::Document& json_data) {
    std::string version = "1.0"; // 您可以根据需要设置版本
    persistence_.WriteWalLog(operation_type, json_data, version); // 将 version 传递给 writeWALLog 方法
}

void VectorDatabase::WriteWalLogWithId(uint64_t log_id, const std::string& data) {
    std::string operation_type = "upsert"; // 默认 operation_type 为 upsert
    std::string version = "1.0"; // 您可以根据需要设置版本
    persistence_.WriteWalRawLog(log_id, operation_type, data, version); // 调用 persistence_ 的 writeWALRawLog 方法
}

auto VectorDatabase::GetIndexTypeFromRequest(const rapidjson::Document& json_request) -> IndexFactory::IndexType {
    // 获取请求参数中的索引类型
    if (json_request.HasMember(REQUEST_INDEX_TYPE) && json_request[REQUEST_INDEX_TYPE].IsString()) {
        std::string index_type_str = json_request[REQUEST_INDEX_TYPE].GetString();
        if (index_type_str == INDEX_TYPE_FLAT) {
            return IndexFactory::IndexType::FLAT;
        } if (index_type_str == INDEX_TYPE_HNSW) {
            return IndexFactory::IndexType::HNSW;
        } else if (index_type_str == INDEX_TYPE_SQ8) {
            return IndexFactory::IndexType::SQ8;
        } else if (index_type_str == INDEX_TYPE_SQ4) {
            return IndexFactory::IndexType::SQ4;
        } else if (index_type_str == INDEX_TYPE_IP_FLAT) {
            return IndexFactory::IndexType::IP_FLAT;
        } else if (index_type_str == INDEX_TYPE_IP_SQ8) {
            return IndexFactory::IndexType::IP_SQ8;
        } else if (index_type_str == INDEX_TYPE_LAYERED_FLAT) {
            return IndexFactory::IndexType::LAYERED_FLAT;
        } else if (index_type_str == INDEX_TYPE_LAYERED_SQ8) {
            return IndexFactory::IndexType::LAYERED_SQ8;
        }
    }
    return IndexFactory::IndexType::UNKNOWN; // 返回UNKNOWN值
}

void VectorDatabase::Upsert(uint64_t id, const rapidjson::Document &data,
                            vectordb::IndexFactory::IndexType index_type) {
  // 检查标量存储中是否存在给定ID的向量
  rapidjson::Document existing_data;  // 修改为驼峰命名
  try {
    existing_data = scalar_storage_.GetScalar(id);
  } catch (const std::runtime_error &e) {
    // 向量不存在，继续执行插入操作
  }

  // 如果存在现有向量，则从索引中删除它
  if (existing_data.IsObject()) {  // 使用IsObject()检查existingData是否为空
    std::vector<float> existing_vector(existing_data["vectors"].Size());  // 从JSON数据中提取vectors字段
    for (rapidjson::SizeType i = 0; i < existing_data["vectors"].Size(); ++i) {
      existing_vector[i] = existing_data["vectors"][i].GetFloat();
    }

    void *index = IndexFactory::Instance().GetIndex(index_type);
    switch (index_type) {
      case IndexFactory::IndexType::FLAT:
      case IndexFactory::IndexType::SQ8:
      case IndexFactory::IndexType::SQ4:
      case IndexFactory::IndexType::IP_FLAT:
      case IndexFactory::IndexType::IP_SQ8: {
        auto *faiss_index = static_cast<FaissIndex *>(index);
        faiss_index->RemoveVectors({static_cast<int64_t>(id)});  // 将id转换为long类型
        break;
      }
      case IndexFactory::IndexType::HNSW: {
        auto *hnsw_index = static_cast<HNSWLibIndex *>(index);
        hnsw_index->RemoveVectors({static_cast<int64_t>(id)});
        break;
      }
      case IndexFactory::IndexType::LAYERED_FLAT:
      case IndexFactory::IndexType::LAYERED_SQ8: {
        auto *layered = static_cast<LayeredIndex *>(index);
        layered->RemoveVectors({static_cast<int64_t>(id)});
        break;
      }
      default:
        break;
    }
  }

  // 将新向量插入索引
  std::vector<float> new_vector(data["vectors"].Size());  // 从JSON数据中提取vectors字段
  for (rapidjson::SizeType i = 0; i < data["vectors"].Size(); ++i) {
    new_vector[i] = data["vectors"][i].GetFloat();
  }

  void *index = IndexFactory::Instance().GetIndex(index_type);
  switch (index_type) {
    case IndexFactory::IndexType::FLAT:
    case IndexFactory::IndexType::SQ8:
    case IndexFactory::IndexType::SQ4:
    case IndexFactory::IndexType::IP_FLAT:
    case IndexFactory::IndexType::IP_SQ8: {
      auto *faiss_index = static_cast<FaissIndex *>(index);
      faiss_index->InsertVectors(new_vector, static_cast<int64_t>(id));
      break;
    }
    case IndexFactory::IndexType::HNSW: {
      auto *hnsw_index = static_cast<HNSWLibIndex *>(index);
      hnsw_index->InsertVectors(new_vector, static_cast<int64_t>(id));
      break;
    }
    case IndexFactory::IndexType::LAYERED_FLAT:
    case IndexFactory::IndexType::LAYERED_SQ8: {
      auto *layered = static_cast<LayeredIndex *>(index);
      layered->InsertVectors(new_vector, static_cast<int64_t>(id));
      break;
    }
    default:
      break;
  }

  global_logger->debug("try add new filter");  // 添加打印信息
  // 检查客户写入的数据中是否有 int 类型的 JSON 字段
  auto *filter_index = static_cast<FilterIndex *>(IndexFactory::Instance().GetIndex(IndexFactory::IndexType::FILTER));
  for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
    std::string field_name = it->name.GetString();
    global_logger->debug("try filter member {} {}", it->value.IsInt(), field_name);  // 添加打印信息
    if (it->value.IsInt() && field_name != "id") {                                   // 过滤名称为 "id" 的字段
      int64_t field_value = it->value.GetInt64();
      int64_t *old_field_value_p = nullptr;
      // 如果存在现有向量，则从 FilterIndex 中更新 int 类型字段
      if (existing_data.IsObject()) {
        old_field_value_p = static_cast<int64_t *>(malloc(sizeof(int64_t)));
        *old_field_value_p = existing_data[field_name.c_str()].GetInt64();
      }
      filter_index->UpdateIntFieldFilter(field_name, old_field_value_p, field_value, id);
      delete old_field_value_p;
    }
    // 字符串字段过滤
    if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
      std::string field_value = it->value.GetString();
      std::string* old_str_value_p = nullptr;
      if (existing_data.IsObject() && existing_data.HasMember(field_name.c_str()) &&
          existing_data[field_name.c_str()].IsString()) {
        old_str_value_p = new std::string(existing_data[field_name.c_str()].GetString());
      }
      filter_index->UpdateStringFieldFilter(field_name, old_str_value_p, field_value, id);
      delete old_str_value_p;
    }
  }

  // 更新标量存储中的向量
  scalar_storage_.InsertScalar(id, data);

  // FullTextIndex: 自动索引 string 字段
  for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
    std::string field_name = it->name.GetString();
    if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
      fulltext_index_.AddDocument(field_name, it->value.GetString(), id);
    }
  }

  // TTLManager: 检测 TTL 字段
  if (data.HasMember(REQUEST_TTL) && data[REQUEST_TTL].IsInt64()) {
    ttl_manager_.SetTTL(id, data[REQUEST_TTL].GetInt64());
  }
}

void VectorDatabase::BatchUpsert(const std::vector<uint64_t>& ids,
                                 const std::vector<rapidjson::Document>& datas,
                                 vectordb::IndexFactory::IndexType index_type) {
  if (ids.empty() || ids.size() != datas.size()) {
    global_logger->error("BatchUpsert: invalid input, ids size={}, datas size={}", ids.size(), datas.size());
    return;
  }

  size_t n = ids.size();

  // 1. 查询已有数据，收集需要删除的旧向量 ID
  std::vector<int64_t> old_ids_to_remove;
  std::vector<rapidjson::Document> existing_datas(n);
  for (size_t i = 0; i < n; ++i) {
    try {
      existing_datas[i] = scalar_storage_.GetScalar(ids[i]);
    } catch (const std::runtime_error&) {
      // 向量不存在，无需删除
    }
    if (existing_datas[i].IsObject() && existing_datas[i].HasMember("vectors")) {
      old_ids_to_remove.push_back(static_cast<int64_t>(ids[i]));
    }
  }

  // 2. 从索引中批量删除旧向量
  if (!old_ids_to_remove.empty()) {
    void* index = IndexFactory::Instance().GetIndex(index_type);
    switch (index_type) {
      case IndexFactory::IndexType::FLAT:
      case IndexFactory::IndexType::SQ8:
      case IndexFactory::IndexType::SQ4:
      case IndexFactory::IndexType::IP_FLAT:
      case IndexFactory::IndexType::IP_SQ8: {
        auto* faiss_index = static_cast<FaissIndex*>(index);
        faiss_index->RemoveVectors(old_ids_to_remove);
        break;
      }
      case IndexFactory::IndexType::HNSW: {
        auto* hnsw_index = static_cast<HNSWLibIndex*>(index);
        hnsw_index->RemoveVectors(old_ids_to_remove);
        break;
      }
      case IndexFactory::IndexType::LAYERED_FLAT:
      case IndexFactory::IndexType::LAYERED_SQ8: {
        auto* layered = static_cast<LayeredIndex*>(index);
        layered->RemoveVectors(old_ids_to_remove);
        break;
      }
      default:
        break;
    }
  }

  // 3. 收集所有新向量并批量插入索引
  int dim = 0;
  std::vector<float> all_vectors;
  std::vector<int64_t> all_labels;
  for (size_t i = 0; i < n; ++i) {
    if (!datas[i].HasMember("vectors") || !datas[i]["vectors"].IsArray()) {
      global_logger->warn("BatchUpsert: item {} missing vectors, skipping", i);
      continue;
    }
    if (dim == 0) {
      dim = datas[i]["vectors"].Size();
    }
    for (rapidjson::SizeType j = 0; j < datas[i]["vectors"].Size(); ++j) {
      all_vectors.push_back(datas[i]["vectors"][j].GetFloat());
    }
    all_labels.push_back(static_cast<int64_t>(ids[i]));
  }

  if (!all_vectors.empty() && !all_labels.empty()) {
    void* index = IndexFactory::Instance().GetIndex(index_type);
    switch (index_type) {
      case IndexFactory::IndexType::FLAT:
      case IndexFactory::IndexType::SQ8:
      case IndexFactory::IndexType::SQ4:
      case IndexFactory::IndexType::IP_FLAT:
      case IndexFactory::IndexType::IP_SQ8: {
        auto* faiss_index = static_cast<FaissIndex*>(index);
        faiss_index->BatchInsertVectors(all_vectors, static_cast<int>(all_labels.size()), all_labels);
        break;
      }
      case IndexFactory::IndexType::HNSW: {
        auto* hnsw_index = static_cast<HNSWLibIndex*>(index);
        hnsw_index->BatchInsertVectors(all_vectors, static_cast<int>(all_labels.size()), all_labels);
        break;
      }
      case IndexFactory::IndexType::LAYERED_FLAT:
      case IndexFactory::IndexType::LAYERED_SQ8: {
        auto* layered = static_cast<LayeredIndex*>(index);
        layered->BatchInsertVectors(all_vectors, static_cast<int>(all_labels.size()), all_labels);
        break;
      }
      default:
        break;
    }
  }

  // 4. 更新 FilterIndex
  auto* filter_index = static_cast<FilterIndex*>(IndexFactory::Instance().GetIndex(IndexFactory::IndexType::FILTER));
  for (size_t i = 0; i < n; ++i) {
    for (auto it = datas[i].MemberBegin(); it != datas[i].MemberEnd(); ++it) {
      std::string field_name = it->name.GetString();
      if (it->value.IsInt() && field_name != "id") {
        int64_t field_value = it->value.GetInt64();
        int64_t* old_field_value_p = nullptr;
        if (existing_datas[i].IsObject() && existing_datas[i].HasMember(field_name.c_str())) {
          old_field_value_p = static_cast<int64_t*>(malloc(sizeof(int64_t)));
          *old_field_value_p = existing_datas[i][field_name.c_str()].GetInt64();
        }
        filter_index->UpdateIntFieldFilter(field_name, old_field_value_p, field_value, ids[i]);
        delete old_field_value_p;
      }
      // 字符串字段过滤
      if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
        std::string field_value = it->value.GetString();
        std::string* old_str_value_p = nullptr;
        if (existing_datas[i].IsObject() && existing_datas[i].HasMember(field_name.c_str()) &&
            existing_datas[i][field_name.c_str()].IsString()) {
          old_str_value_p = new std::string(existing_datas[i][field_name.c_str()].GetString());
        }
        filter_index->UpdateStringFieldFilter(field_name, old_str_value_p, field_value, ids[i]);
        delete old_str_value_p;
      }
    }
  }

  // 5. 批量写入标量存储
  scalar_storage_.BatchInsertScalar(ids, datas);

  // 6. 更新 FullTextIndex 和 TTLManager
  for (size_t i = 0; i < n; ++i) {
    for (auto it = datas[i].MemberBegin(); it != datas[i].MemberEnd(); ++it) {
      std::string field_name = it->name.GetString();
      if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
        fulltext_index_.AddDocument(field_name, it->value.GetString(), ids[i]);
      }
    }
    if (datas[i].HasMember(REQUEST_TTL) && datas[i][REQUEST_TTL].IsInt64()) {
      ttl_manager_.SetTTL(ids[i], datas[i][REQUEST_TTL].GetInt64());
    }
  }

  global_logger->info("BatchUpsert completed: {} vectors inserted", n);
}

auto VectorDatabase::Query(uint64_t id) -> rapidjson::Document {  // 添加query函数实现
  return scalar_storage_.GetScalar(id);
}


auto VectorDatabase::Search(const rapidjson::Document& json_request) -> std::pair<std::vector<int64_t>, std::vector<float>> {
    // 从 JSON 请求中获取查询参数
    std::vector<float> query;
    for (const auto& q : json_request[REQUEST_VECTORS].GetArray()) {
        query.push_back(q.GetFloat());
    }
    int k = json_request[REQUEST_K].GetInt();

    // 获取请求参数中的索引类型
    IndexFactory::IndexType index_type = IndexFactory::IndexType::UNKNOWN;
    if (json_request.HasMember(REQUEST_INDEX_TYPE) && json_request[REQUEST_INDEX_TYPE].IsString()) {
        std::string index_type_str = json_request[REQUEST_INDEX_TYPE].GetString();
        if (index_type_str == INDEX_TYPE_FLAT) {
            index_type = IndexFactory::IndexType::FLAT;
        } else if (index_type_str == INDEX_TYPE_HNSW) {
            index_type = IndexFactory::IndexType::HNSW;
        } else if (index_type_str == INDEX_TYPE_SQ8) {
            index_type = IndexFactory::IndexType::SQ8;
        } else if (index_type_str == INDEX_TYPE_SQ4) {
            index_type = IndexFactory::IndexType::SQ4;
        } else if (index_type_str == INDEX_TYPE_IP_FLAT) {
            index_type = IndexFactory::IndexType::IP_FLAT;
        } else if (index_type_str == INDEX_TYPE_IP_SQ8) {
            index_type = IndexFactory::IndexType::IP_SQ8;
        } else if (index_type_str == INDEX_TYPE_LAYERED_FLAT) {
            index_type = IndexFactory::IndexType::LAYERED_FLAT;
        } else if (index_type_str == INDEX_TYPE_LAYERED_SQ8) {
            index_type = IndexFactory::IndexType::LAYERED_SQ8;
        }
    }

    // 检查请求中是否包含 filter 参数
    roaring_bitmap_t* filter_bitmap = nullptr;
    auto* filter_index = static_cast<FilterIndex*>(IndexFactory::Instance().GetIndex(IndexFactory::IndexType::FILTER));

    // 多条件 AND（filters 数组）
    if (json_request.HasMember("filters") && json_request["filters"].IsArray()) {
        const auto& filters = json_request["filters"].GetArray();
        for (rapidjson::SizeType fi = 0; fi < filters.Size(); ++fi) {
            const auto& filter = filters[fi];
            if (!filter.IsObject() || !filter.HasMember("fieldName") || !filter.HasMember("op") || !filter.HasMember("value"))
                continue;

            std::string field_name = filter["fieldName"].GetString();
            std::string op_str = filter["op"].GetString();
            FilterIndex::Operation op = ParseFilterOp(op_str);

            roaring_bitmap_t* condition_bitmap = roaring_bitmap_create();
            std::string field_type = filter.HasMember("fieldType") ? std::string(filter["fieldType"].GetString()) : "int";

            if (field_type == "string" && filter["value"].IsString()) {
                std::string value = filter["value"].GetString();
                filter_index->GetStringFieldFilterBitmap(field_name, op, value, condition_bitmap);
            } else if (filter["value"].IsInt64()) {
                int64_t value = filter["value"].GetInt64();
                filter_index->GetIntFieldFilterBitmap(field_name, op, value, condition_bitmap);
            }

            if (filter_bitmap == nullptr) {
                filter_bitmap = condition_bitmap;
            } else {
                roaring_bitmap_and_inplace(filter_bitmap, condition_bitmap);
                roaring_bitmap_free(condition_bitmap);
            }
        }
    } else if (json_request.HasMember("filter") && json_request["filter"].IsObject()) {
        // 单条件（向后兼容）
        const auto& filter = json_request["filter"];
        std::string field_name = filter["fieldName"].GetString();
        std::string op_str = filter["op"].GetString();
        FilterIndex::Operation op = ParseFilterOp(op_str);

        filter_bitmap = roaring_bitmap_create();
        if (filter["value"].IsString()) {
            std::string value = filter["value"].GetString();
            filter_index->GetStringFieldFilterBitmap(field_name, op, value, filter_bitmap);
        } else {
            int64_t value = filter["value"].GetInt64();
            filter_index->GetIntFieldFilterBitmap(field_name, op, value, filter_bitmap);
        }
    }

    // 全文搜索：如果有 fulltext 参数，先做 BM25 搜索，转为 bitmap 过滤
    if (json_request.HasMember(REQUEST_FULLTEXT) && json_request[REQUEST_FULLTEXT].IsObject()) {
        const auto& ft = json_request[REQUEST_FULLTEXT];
        if (ft.HasMember("field") && ft.HasMember("query")) {
            std::string ft_field = ft["field"].GetString();
            std::string ft_query = ft["query"].GetString();
            auto [ft_ids, ft_scores] = fulltext_index_.Search(ft_field, ft_query, 10000);
            roaring_bitmap_t* ft_bitmap = roaring_bitmap_create();
            for (uint64_t ft_id : ft_ids) {
                roaring_bitmap_add(ft_bitmap, static_cast<uint32_t>(ft_id));
            }
            if (filter_bitmap == nullptr) {
                filter_bitmap = ft_bitmap;
            } else {
                roaring_bitmap_and_inplace(filter_bitmap, ft_bitmap);
                roaring_bitmap_free(ft_bitmap);
            }
        }
    }

    // 使用全局 IndexFactory 获取索引对象
    void* index = IndexFactory::Instance().GetIndex(index_type);

    // 根据索引类型初始化索引对象并调用 search_vectors 函数
    std::pair<std::vector<int64_t>, std::vector<float>> results;
    switch (index_type) {
        case IndexFactory::IndexType::FLAT:
        case IndexFactory::IndexType::SQ8:
        case IndexFactory::IndexType::SQ4:
        case IndexFactory::IndexType::IP_FLAT:
        case IndexFactory::IndexType::IP_SQ8: {
            auto* faiss_index = static_cast<FaissIndex*>(index);
            results = faiss_index->SearchVectors(query, k, filter_bitmap); // 将 filter_bitmap 传递给 search_vectors 方法
            break;
        }
        case IndexFactory::IndexType::HNSW: {
            auto* hnsw_index = static_cast<HNSWLibIndex*>(index);
            results = hnsw_index->SearchVectors(query, k, filter_bitmap); // 将 filter_bitmap 传递给 search_vectors 方法
            break;
        }
        case IndexFactory::IndexType::LAYERED_FLAT:
        case IndexFactory::IndexType::LAYERED_SQ8: {
            auto* layered = static_cast<LayeredIndex*>(index);
            results = layered->SearchVectors(query, k, filter_bitmap);
            break;
        }
        // 在此处添加其他索引类型的处理逻辑
        default:
            break;
    }
    delete filter_bitmap;

    // TTL 惰性过滤：移除过期 ID
    for (size_t i = 0; i < results.first.size(); ++i) {
        if (results.first[i] != -1 && ttl_manager_.IsExpired(static_cast<uint64_t>(results.first[i]))) {
            results.first[i] = -1;
        }
    }

    return results;
}
void VectorDatabase::TakeSnapshot() { // 添加 takeSnapshot 方法实现
    persistence_.TakeSnapshot();
}

auto VectorDatabase::GetStartIndexId() const -> int64_t {
    return persistence_.GetId(); // 通过调用 persistence_ 的 GetID 方法获取起始索引 ID
}

void VectorDatabase::CleanExpiredVectors() {
    auto expired_ids = ttl_manager_.GetExpiredIDs();
    for (uint64_t id : expired_ids) {
        ttl_manager_.Remove(id);
    }
    if (!expired_ids.empty()) {
        global_logger->info("Cleaned {} expired vectors", expired_ids.size());
    }
}

}  // namespace vectordb