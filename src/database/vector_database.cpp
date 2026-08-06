#include "database/vector_database.h"
#include "collection/collection_manager.h"
#include <rapidjson/document.h>
#include <cstdint>
#include <vector>
#include "common/constants.h"
#include "common/vector_cfg.h"
#include "database/scalar_storage.h"
#include "index/faiss_index.h"
#include "index/filter_index.h"
#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "index/layered_index.h"
#include "logger/logger.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

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

// 从 JSON 请求中解析 collectionName（默认 "default"）
auto GetCollectionNameFromRequest(const rapidjson::Document& json_request) -> std::string {
    if (json_request.HasMember(REQUEST_COLLECTION_NAME) && json_request[REQUEST_COLLECTION_NAME].IsString()) {
        return json_request[REQUEST_COLLECTION_NAME].GetString();
    }
    return DEFAULT_COLLECTION_NAME;
}
} // anonymous namespace

VectorDatabase::VectorDatabase(const std::string &db_path, const std::string& wal_path) : scalar_storage_(db_path) {
    persistence_.Init(wal_path);
}

void VectorDatabase::ReloadDatabase() {
    global_logger->info("Entering VectorDatabase::reloadDatabase()");

    persistence_.LoadSnapshot();

    std::string operation_type;
    rapidjson::Document json_data;
    persistence_.ReadNextWalLog(&operation_type, &json_data);

    while (!operation_type.empty()) {
        global_logger->info("Operation Type: {}", operation_type);

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        json_data.Accept(writer);
        global_logger->info("Read Line: {}", buffer.GetString());

       if (operation_type == "upsert") {
            std::string collection_name = GetCollectionNameFromRequest(json_data);
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
                        BatchUpsert(collection_name, ids, datas, index_type);
                    }
                }
            } else {
                uint64_t id = json_data[REQUEST_ID].GetUint64();
                IndexFactory::IndexType index_type = GetIndexTypeFromRequest(json_data);
                Upsert(collection_name, id, json_data, index_type);
            }
        }

        rapidjson::Document().Swap(json_data);

        operation_type.clear();
        persistence_.ReadNextWalLog(&operation_type, &json_data);
    }
}

void VectorDatabase::WriteWalLog(const std::string& operation_type, const rapidjson::Document& json_data) {
    std::string version = "1.0";
    persistence_.WriteWalLog(operation_type, json_data, version);
}

void VectorDatabase::WriteWalLogWithId(uint64_t log_id, const std::string& data) {
    std::string operation_type = "upsert";
    std::string version = "1.0";
    persistence_.WriteWalRawLog(log_id, operation_type, data, version);
}

auto VectorDatabase::GetIndexTypeFromRequest(const rapidjson::Document& json_request) -> IndexFactory::IndexType {
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
    return IndexFactory::IndexType::UNKNOWN;
}

void VectorDatabase::Upsert(const std::string& collection_name, uint64_t id, const rapidjson::Document &data,
                            vectordb::IndexFactory::IndexType index_type) {
  auto* coll = CollectionManager::Instance().GetCollection(collection_name);
  if (coll == nullptr) {
    // 自动创建 Collection：从第一条向量数据推断维度
    int dim = data.HasMember("vectors") ? data["vectors"].Size() : 0;
    if (dim <= 0) {
      global_logger->error("Upsert: collection '{}' not found and cannot infer dimension", collection_name);
      return;
    }
    global_logger->info("Auto-creating collection '{}' dim={}", collection_name, dim);
    CollectionManager::Instance().CreateCollection(collection_name, dim, Cfg::Instance().NumData());
    coll = CollectionManager::Instance().GetCollection(collection_name);
    if (coll == nullptr) return;
  }

  // 检查标量存储中是否存在给定ID的向量
  rapidjson::Document existing_data;
  try {
    existing_data = scalar_storage_.GetScalar(collection_name, id);
  } catch (const std::runtime_error &e) {
    // 向量不存在，继续执行插入操作
  }

  // 如果存在现有记录，则从索引中按 ID 删除旧向量（RemoveVectors 接收 ID，无需旧向量数据）
  if (existing_data.IsObject()) {
    void *index = coll->GetIndex(index_type);
    switch (index_type) {
      case IndexFactory::IndexType::FLAT:
      case IndexFactory::IndexType::SQ8:
      case IndexFactory::IndexType::SQ4:
      case IndexFactory::IndexType::IP_FLAT:
      case IndexFactory::IndexType::IP_SQ8: {
        auto *faiss_index = static_cast<FaissIndex *>(index);
        faiss_index->RemoveVectors({static_cast<int64_t>(id)});
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
      case IndexFactory::IndexType::GARDEN_HNSW: {
        auto *garden = static_cast<GardenIndex *>(index);
        garden->RemoveVectors({static_cast<int64_t>(id)});
        break;
      }
      default:
        break;
    }
  }

  // 将新向量插入索引
  std::vector<float> new_vector(data["vectors"].Size());
  for (rapidjson::SizeType i = 0; i < data["vectors"].Size(); ++i) {
    new_vector[i] = data["vectors"][i].GetFloat();
  }

  void *index = coll->GetIndex(index_type);
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
    case IndexFactory::IndexType::GARDEN_HNSW: {
      auto *garden = static_cast<GardenIndex *>(index);
      std::map<std::string, int64_t> int_fields;
      std::map<std::string, std::string> str_fields;
      for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
        std::string fname = it->name.GetString();
        if (fname == "vectors" || fname == "id") continue;
        if (it->value.IsInt64()) int_fields[fname] = it->value.GetInt64();
        else if (it->value.IsString()) str_fields[fname] = it->value.GetString();
      }
      for (auto& [fn, fv] : str_fields) {
        if (!garden->HasDiscreteSubgraph(fn)) garden->RegisterDiscreteField(fn);
      }
      garden->Insert(new_vector, static_cast<int64_t>(id), int_fields, str_fields);
      break;
    }
    default:
      break;
  }

  global_logger->debug("try add new filter");
  auto *filter_index = static_cast<FilterIndex *>(coll->GetIndex(IndexFactory::IndexType::FILTER));
  for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
    std::string field_name = it->name.GetString();
    global_logger->debug("try filter member {} {}", it->value.IsInt(), field_name);
    if (it->value.IsInt() && field_name != "id") {
      int64_t field_value = it->value.GetInt64();
      int64_t *old_field_value_p = nullptr;
      if (existing_data.IsObject()) {
        old_field_value_p = static_cast<int64_t *>(malloc(sizeof(int64_t)));
        *old_field_value_p = existing_data[field_name.c_str()].GetInt64();
      }
      filter_index->UpdateIntFieldFilter(field_name, old_field_value_p, field_value, id);
      delete old_field_value_p;
    }
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

  scalar_storage_.InsertScalar(collection_name, id, data);
  scalar_storage_.InsertVector(collection_name, id, new_vector);

  // FullTextIndex: 自动索引 string 字段
  for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
    std::string field_name = it->name.GetString();
    if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
      coll->fulltext_index.AddDocument(field_name, it->value.GetString(), id);
    }
  }

  // TTLManager: 检测 TTL 字段
  if (data.HasMember(REQUEST_TTL) && data[REQUEST_TTL].IsInt64()) {
    coll->ttl_manager.SetTTL(id, data[REQUEST_TTL].GetInt64());
  }
}

void VectorDatabase::BatchUpsert(const std::string& collection_name,
                                 const std::vector<uint64_t>& ids,
                                 const std::vector<rapidjson::Document>& datas,
                                 vectordb::IndexFactory::IndexType index_type) {
  auto* coll = CollectionManager::Instance().GetCollection(collection_name);
  if (coll == nullptr) {
    // 自动创建 Collection：从第一条有效向量数据推断维度
    int dim = 0;
    for (const auto& doc : datas) {
      if (doc.HasMember("vectors") && doc["vectors"].IsArray()) {
        dim = doc["vectors"].Size();
        break;
      }
    }
    if (dim <= 0) {
      global_logger->error("BatchUpsert: collection '{}' not found and cannot infer dimension", collection_name);
      return;
    }
    global_logger->info("Auto-creating collection '{}' dim={}", collection_name, dim);
    CollectionManager::Instance().CreateCollection(collection_name, dim, Cfg::Instance().NumData());
    coll = CollectionManager::Instance().GetCollection(collection_name);
    if (coll == nullptr) return;
  }

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
      existing_datas[i] = scalar_storage_.GetScalar(collection_name, ids[i]);
    } catch (const std::runtime_error&) {
      // 向量不存在，无需删除
    }
    if (existing_datas[i].IsObject()) {
      old_ids_to_remove.push_back(static_cast<int64_t>(ids[i]));
    }
  }

  // 2. 从索引中批量删除旧向量
  if (!old_ids_to_remove.empty()) {
    void* index = coll->GetIndex(index_type);
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
      case IndexFactory::IndexType::GARDEN_HNSW: {
        auto* garden = static_cast<GardenIndex*>(index);
        garden->RemoveVectors(old_ids_to_remove);
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
    void* index = coll->GetIndex(index_type);
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
      case IndexFactory::IndexType::GARDEN_HNSW: {
        auto* garden = static_cast<GardenIndex*>(index);
        for (size_t i = 0; i < all_labels.size(); ++i) {
          std::vector<float> vec(all_vectors.begin() + i * dim, all_vectors.begin() + (i + 1) * dim);
          std::map<std::string, int64_t> int_fields;
          std::map<std::string, std::string> str_fields;
          for (auto it = datas[i].MemberBegin(); it != datas[i].MemberEnd(); ++it) {
            std::string fname = it->name.GetString();
            if (fname == "vectors" || fname == "id") continue;
            if (it->value.IsInt64()) int_fields[fname] = it->value.GetInt64();
            else if (it->value.IsString()) str_fields[fname] = it->value.GetString();
          }
          for (auto& [fn, fv] : str_fields) {
            if (!garden->HasDiscreteSubgraph(fn)) garden->RegisterDiscreteField(fn);
          }
          garden->Insert(vec, all_labels[i], int_fields, str_fields);
        }
        break;
      }
      default:
        break;
    }
  }

  // 4. 更新 FilterIndex
  auto* filter_index = static_cast<FilterIndex*>(coll->GetIndex(IndexFactory::IndexType::FILTER));
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

  // 5. 批量写入标量存储 + 向量存储
  scalar_storage_.BatchInsertScalar(collection_name, ids, datas);
  if (!all_vectors.empty() && !all_labels.empty() && dim > 0) {
    std::vector<uint64_t> vec_ids(all_labels.begin(), all_labels.end());
    scalar_storage_.BatchInsertVector(collection_name, vec_ids, all_vectors, dim);
  }

  // 6. 更新 FullTextIndex 和 TTLManager
  for (size_t i = 0; i < n; ++i) {
    for (auto it = datas[i].MemberBegin(); it != datas[i].MemberEnd(); ++it) {
      std::string field_name = it->name.GetString();
      if (it->value.IsString() && field_name != "vectors" && field_name != "id") {
        coll->fulltext_index.AddDocument(field_name, it->value.GetString(), ids[i]);
      }
    }
    if (datas[i].HasMember(REQUEST_TTL) && datas[i][REQUEST_TTL].IsInt64()) {
      coll->ttl_manager.SetTTL(ids[i], datas[i][REQUEST_TTL].GetInt64());
    }
  }

  global_logger->info("BatchUpsert completed: {} vectors inserted into '{}'", n, collection_name);
}

auto VectorDatabase::Query(const std::string& collection_name, uint64_t id) -> rapidjson::Document {
  rapidjson::Document doc = scalar_storage_.GetScalar(collection_name, id);
  if (doc.IsObject()) {
    std::vector<float> vec = scalar_storage_.GetVector(collection_name, id);
    if (!vec.empty()) {
      rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
      rapidjson::Value arr(rapidjson::kArrayType);
      for (float v : vec) {
        arr.PushBack(v, alloc);
      }
      doc.AddMember("vectors", arr, alloc);
    }
  }
  return doc;
}


auto VectorDatabase::Search(const std::string& collection_name, const rapidjson::Document& json_request) -> std::pair<std::vector<int64_t>, std::vector<float>> {
    auto* coll = CollectionManager::Instance().GetCollection(collection_name);
    if (coll == nullptr) {
        global_logger->error("Search: collection '{}' not found", collection_name);
        return {{}, {}};
    }

    std::vector<float> query;
    for (const auto& q : json_request[REQUEST_VECTORS].GetArray()) {
        query.push_back(q.GetFloat());
    }
    int k = json_request[REQUEST_K].GetInt();

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
        } else if (index_type_str == INDEX_TYPE_GARDEN_HNSW) {
            index_type = IndexFactory::IndexType::GARDEN_HNSW;
        }
    }

    // 解析 excludeIds 黑名单（Selector 角色）：GARDEN 路径作 selector_bitmap，通用路径做后过滤
    roaring_bitmap_t* exclude_bitmap = nullptr;
    if (json_request.HasMember(REQUEST_EXCLUDE_IDS) && json_request[REQUEST_EXCLUDE_IDS].IsArray()) {
        exclude_bitmap = roaring_bitmap_create();
        for (const auto& v : json_request[REQUEST_EXCLUDE_IDS].GetArray()) {
            if (v.IsUint64()) {
                roaring_bitmap_add(exclude_bitmap, static_cast<uint32_t>(v.GetUint64()));
            }
        }
    }

    // ===== GARDEN_HNSW 独立路径：多子图 + Pruner/Grafter/Selector 分级 =====
    if (index_type == IndexFactory::IndexType::GARDEN_HNSW) {
        auto* garden = static_cast<GardenIndex*>(coll->GetIndex(index_type));
        if (garden == nullptr) {
            global_logger->error("Search: GARDEN_HNSW index not initialized for '{}'", collection_name);
            if (exclude_bitmap != nullptr) roaring_bitmap_free(exclude_bitmap);
            return {{}, {}};
        }

        auto* filter_index = static_cast<FilterIndex*>(coll->GetIndex(IndexFactory::IndexType::FILTER));
        std::vector<GardenFilter> garden_filters;

        // 解析 filter 条件（支持 filters 数组和单 filter 对象）
        auto parse_one_filter = [&](const rapidjson::Value& filter) {
            if (!filter.IsObject() || !filter.HasMember("fieldName") || !filter.HasMember("op"))
                return;
            std::string op_str = filter["op"].GetString();
            // range op 用 min/max，其它 op 用 value
            if (op_str != "range" && !filter.HasMember("value"))
                return;
            std::string field_name = filter["fieldName"].GetString();
            FilterIndex::Operation op = ParseFilterOp(op_str);
            std::string field_type = filter.HasMember("fieldType")
                ? std::string(filter["fieldType"].GetString()) : "int";

            roaring_bitmap_t* cond_bitmap = roaring_bitmap_create();
            GardenFilter gf;
            gf.field = field_name;

            if (field_type == "string" && filter["value"].IsString()) {
                std::string value = filter["value"].GetString();
                filter_index->GetStringFieldFilterBitmap(field_name, op, value, cond_bitmap);
                gf.discrete_values.push_back(value);
            } else if (op_str == "range" && filter.HasMember("min") && filter.HasMember("max")) {
                // 区间查询：min <= field <= max，bitmap = (>=min) AND (<=max)
                int64_t range_min = filter["min"].GetInt64();
                int64_t range_max = filter["max"].GetInt64();
                roaring_bitmap_t* ge_bm = roaring_bitmap_create();
                filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::GREATER_EQUAL,
                                                      range_min, ge_bm);
                filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::LESS_EQUAL,
                                                      range_max, cond_bitmap);
                roaring_bitmap_and_inplace(cond_bitmap, ge_bm);
                roaring_bitmap_free(ge_bm);
                gf.has_range = true;
                gf.range_min = range_min;
                gf.range_max = range_max;
            } else if (filter["value"].IsInt64()) {
                int64_t value = filter["value"].GetInt64();
                filter_index->GetIntFieldFilterBitmap(field_name, op, value, cond_bitmap);
                // 单边条件补全 range 另一端，避免 GARDEN 连续 Pruner 桶收集退化
                if (op == FilterIndex::Operation::GREATER_EQUAL ||
                    op == FilterIndex::Operation::GREATER_THAN) {
                    gf.has_range = true;
                    gf.range_min = value;
                    gf.range_max = INT64_MAX;
                } else if (op == FilterIndex::Operation::LESS_EQUAL ||
                           op == FilterIndex::Operation::LESS_THAN) {
                    gf.has_range = true;
                    gf.range_min = INT64_MIN;
                    gf.range_max = value;
                }
            }

            gf.bitmap = cond_bitmap;
            gf.cardinality = roaring_bitmap_get_cardinality(cond_bitmap);
            garden_filters.push_back(gf);
        };

        if (json_request.HasMember("filters") && json_request["filters"].IsArray()) {
            for (const auto& f : json_request["filters"].GetArray()) {
                parse_one_filter(f);
            }
        } else if (json_request.HasMember("filter") && json_request["filter"].IsObject()) {
            parse_one_filter(json_request["filter"]);
        }

        // 全文搜索结果作为额外 Grafter 条件
        if (json_request.HasMember(REQUEST_FULLTEXT) && json_request[REQUEST_FULLTEXT].IsObject()) {
            const auto& ft = json_request[REQUEST_FULLTEXT];
            if (ft.HasMember("field") && ft.HasMember("query")) {
                auto [ft_ids, ft_scores] = coll->fulltext_index.Search(
                    ft["field"].GetString(), ft["query"].GetString(), 10000);
                roaring_bitmap_t* ft_bitmap = roaring_bitmap_create();
                for (uint64_t ft_id : ft_ids) {
                    roaring_bitmap_add(ft_bitmap, static_cast<uint32_t>(ft_id));
                }
                GardenFilter ft_filter;
                ft_filter.field = "_fulltext";
                ft_filter.bitmap = ft_bitmap;
                ft_filter.cardinality = roaring_bitmap_get_cardinality(ft_bitmap);
                garden_filters.push_back(ft_filter);
            }
        }

        auto result = garden->Search(query, k, garden_filters, exclude_bitmap, 50);

        // 清理临时 bitmap
        for (auto& gf : garden_filters) {
            if (gf.bitmap) {
                roaring_bitmap_free(const_cast<roaring_bitmap_t*>(gf.bitmap));
            }
        }
        if (exclude_bitmap != nullptr) {
            roaring_bitmap_free(exclude_bitmap);
        }

        // TTL 惰性过滤：GARDEN 独立路径提前 return，需在此过滤过期 ID
        // （与下方通用路径 712-717 行的 TTL 过滤逻辑对齐）
        for (size_t i = 0; i < result.first.size(); ++i) {
            if (result.first[i] != -1 &&
                coll->ttl_manager.IsExpired(static_cast<uint64_t>(result.first[i]))) {
                result.first[i] = -1;
            }
        }

        return result;
    }

    roaring_bitmap_t* filter_bitmap = nullptr;
    auto* filter_index = static_cast<FilterIndex*>(coll->GetIndex(IndexFactory::IndexType::FILTER));

    // 多条件 AND（filters 数组）
    if (json_request.HasMember("filters") && json_request["filters"].IsArray()) {
        const auto& filters = json_request["filters"].GetArray();
        for (rapidjson::SizeType fi = 0; fi < filters.Size(); ++fi) {
            const auto& filter = filters[fi];
            if (!filter.IsObject() || !filter.HasMember("fieldName") || !filter.HasMember("op"))
                continue;
            std::string field_name = filter["fieldName"].GetString();
            std::string op_str = filter["op"].GetString();
            if (op_str != "range" && !filter.HasMember("value"))
                continue;

            FilterIndex::Operation op = ParseFilterOp(op_str);

            roaring_bitmap_t* condition_bitmap = roaring_bitmap_create();
            std::string field_type = filter.HasMember("fieldType") ? std::string(filter["fieldType"].GetString()) : "int";

            if (field_type == "string" && filter["value"].IsString()) {
                std::string value = filter["value"].GetString();
                filter_index->GetStringFieldFilterBitmap(field_name, op, value, condition_bitmap);
            } else if (op_str == "range" && filter.HasMember("min") && filter.HasMember("max")) {
                int64_t range_min = filter["min"].GetInt64();
                int64_t range_max = filter["max"].GetInt64();
                roaring_bitmap_t* ge_bm = roaring_bitmap_create();
                filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::GREATER_EQUAL, range_min, ge_bm);
                filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::LESS_EQUAL, range_max, condition_bitmap);
                roaring_bitmap_and_inplace(condition_bitmap, ge_bm);
                roaring_bitmap_free(ge_bm);
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
        const auto& filter = json_request["filter"];
        std::string field_name = filter["fieldName"].GetString();
        std::string op_str = filter["op"].GetString();
        FilterIndex::Operation op = ParseFilterOp(op_str);

        filter_bitmap = roaring_bitmap_create();
        if (op_str == "range" && filter.HasMember("min") && filter.HasMember("max")) {
            int64_t range_min = filter["min"].GetInt64();
            int64_t range_max = filter["max"].GetInt64();
            roaring_bitmap_t* ge_bm = roaring_bitmap_create();
            filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::GREATER_EQUAL, range_min, ge_bm);
            filter_index->GetIntFieldFilterBitmap(field_name, FilterIndex::Operation::LESS_EQUAL, range_max, filter_bitmap);
            roaring_bitmap_and_inplace(filter_bitmap, ge_bm);
            roaring_bitmap_free(ge_bm);
        } else if (filter["value"].IsString()) {
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
            auto [ft_ids, ft_scores] = coll->fulltext_index.Search(ft_field, ft_query, 10000);
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

    void* index = coll->GetIndex(index_type);

    std::pair<std::vector<int64_t>, std::vector<float>> results;
    switch (index_type) {
        case IndexFactory::IndexType::FLAT:
        case IndexFactory::IndexType::SQ8:
        case IndexFactory::IndexType::SQ4:
        case IndexFactory::IndexType::IP_FLAT:
        case IndexFactory::IndexType::IP_SQ8: {
            auto* faiss_index = static_cast<FaissIndex*>(index);
            results = faiss_index->SearchVectors(query, k, filter_bitmap);
            break;
        }
        case IndexFactory::IndexType::HNSW: {
            auto* hnsw_index = static_cast<HNSWLibIndex*>(index);
            results = hnsw_index->SearchVectors(query, k, filter_bitmap);
            break;
        }
        case IndexFactory::IndexType::LAYERED_FLAT:
        case IndexFactory::IndexType::LAYERED_SQ8: {
            auto* layered = static_cast<LayeredIndex*>(index);
            results = layered->SearchVectors(query, k, filter_bitmap);
            break;
        }
        default:
            break;
    }
    delete filter_bitmap;

    // TTL 惰性过滤：移除过期 ID
    for (size_t i = 0; i < results.first.size(); ++i) {
        if (results.first[i] != -1 && coll->ttl_manager.IsExpired(static_cast<uint64_t>(results.first[i]))) {
            results.first[i] = -1;
        }
    }

    // excludeIds 黑名单后过滤（通用路径无 selector_bitmap 参数，后过滤移除）
    if (exclude_bitmap != nullptr) {
        for (size_t i = 0; i < results.first.size(); ++i) {
            if (results.first[i] != -1 &&
                roaring_bitmap_contains(exclude_bitmap, static_cast<uint32_t>(results.first[i]))) {
                results.first[i] = -1;
            }
        }
        roaring_bitmap_free(exclude_bitmap);
    }

    return results;
}
void VectorDatabase::TakeSnapshot() {
    persistence_.TakeSnapshot();
}

auto VectorDatabase::GetStartIndexId() const -> int64_t {
    return persistence_.GetId();
}

void VectorDatabase::CleanExpiredVectors() {
    // 遍历所有 Collection，清理过期向量
    auto names = CollectionManager::Instance().ListCollections();
    for (const auto& name : names) {
        auto* coll = CollectionManager::Instance().GetCollection(name);
        if (coll == nullptr) continue;
        auto expired_ids = coll->ttl_manager.GetExpiredIDs();
        for (uint64_t id : expired_ids) {
            coll->ttl_manager.Remove(id);
        }
        if (!expired_ids.empty()) {
            global_logger->info("Cleaned {} expired vectors from collection '{}'", expired_ids.size(), name);
        }
    }
}

}  // namespace vectordb
