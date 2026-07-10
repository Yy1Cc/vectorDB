#ifndef SCALART_STORAGE_H
#define SCALART_STORAGE_H

#include <rocksdb/db.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/cache.h>
#include <string>
#include <vector>
#include <rapidjson/document.h>

namespace vectordb {

class ScalarStorage {
public:
    explicit ScalarStorage(const std::string& db_path);
    ~ScalarStorage();

    // ---- Scalar operations (scalar CF, binary TLV format, excludes "vectors") ----
    void InsertScalar(const std::string& collection_name, uint64_t id, const rapidjson::Document& data);
    void BatchInsertScalar(const std::string& collection_name, const std::vector<uint64_t>& ids, const std::vector<rapidjson::Document>& datas);
    auto GetScalar(const std::string& collection_name, uint64_t id) -> rapidjson::Document;

    // ---- Vector operations (vector CF, raw float32 binary) ----
    void InsertVector(const std::string& collection_name, uint64_t id, const std::vector<float>& vec);
    void BatchInsertVector(const std::string& collection_name, const std::vector<uint64_t>& ids, const std::vector<float>& flat_vectors, int dim);
    auto GetVector(const std::string& collection_name, uint64_t id) -> std::vector<float>;

private:
    rocksdb::DB* db_;
    rocksdb::ColumnFamilyHandle* default_cf_handle_ = nullptr;
    rocksdb::ColumnFamilyHandle* scalar_cf_handle_   = nullptr;
    rocksdb::ColumnFamilyHandle* vector_cf_handle_   = nullptr;

    static auto MakeKey(const std::string& collection_name, uint64_t id) -> std::string;

    // Binary TLV serialization for scalar fields (skips "vectors" member)
    static auto SerializeScalar(const rapidjson::Document& data) -> std::string;
    static auto DeserializeScalar(const std::string& binary) -> rapidjson::Document;

    // Raw float32 serialization for vectors
    static auto SerializeVector(const float* data, size_t count) -> std::string;
    static auto DeserializeVector(const std::string& binary) -> std::vector<float>;

    // RocksDB options configuration
    static void ConfigureDBOptions(rocksdb::DBOptions& opts);
    static void ConfigureCFOptions(rocksdb::ColumnFamilyOptions& opts);
};

}  // namespace vectordb

#endif
