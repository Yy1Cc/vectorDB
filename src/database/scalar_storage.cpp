#include "database/scalar_storage.h"
#include "logger/logger.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rapidjson/document.h>
#include <cstring>
#include <vector>

namespace vectordb {

// ============================================================
// Binary TLV helpers (little-endian, no external dependency)
// ============================================================

namespace {

constexpr uint8_t  kScalarMagic   = 0xDB;
constexpr uint8_t  kScalarVersion = 0x01;

// Field type codes
constexpr uint8_t  kTypeInt64   = 0;
constexpr uint8_t  kTypeUint64  = 1;
constexpr uint8_t  kTypeString  = 2;
constexpr uint8_t  kTypeDouble  = 3;
constexpr uint8_t  kTypeBool    = 4;

// LE write helpers
inline void PutU8 (std::string& buf, uint8_t  v) { buf.push_back(static_cast<char>(v)); }
inline void PutU16(std::string& buf, uint16_t v) { buf.append(reinterpret_cast<const char*>(&v), 2); }
inline void PutU32(std::string& buf, uint32_t v) { buf.append(reinterpret_cast<const char*>(&v), 4); }
inline void PutU64(std::string& buf, uint64_t v) { buf.append(reinterpret_cast<const char*>(&v), 8); }
inline void PutF64(std::string& buf, double   v) { buf.append(reinterpret_cast<const char*>(&v), 8); }

// LE read helpers
inline uint8_t  GetU8 (const char* p) { return static_cast<uint8_t>(*p); }
inline uint16_t GetU16(const char* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t GetU32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t GetU64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline double   GetF64(const char* p) { double   v; std::memcpy(&v, p, 8); return v; }

} // anonymous namespace

// ============================================================
// RocksDB Options Configuration
// ============================================================

void ScalarStorage::ConfigureDBOptions(rocksdb::DBOptions& opts) {
    opts.create_if_missing = true;
    opts.create_missing_column_families = true;  // auto-create scalar/vector CFs

    // Background threads: parallel flush + compaction
    opts.max_background_jobs       = 12;
    opts.max_background_flushes    = 4;
    opts.max_background_compactions = 8;

    // Keep WAL for durability (scalar storage is the source of truth for point lookups)
    opts.max_total_wal_size        = 512ULL * 1024 * 1024;  // 512MB WAL
}

void ScalarStorage::ConfigureCFOptions(rocksdb::ColumnFamilyOptions& opts) {
    // --- Write buffer: reduce flush frequency for batch workloads ---
    opts.write_buffer_size             = 256ULL * 1024 * 1024;  // 256MB memtable
    opts.max_write_buffer_number       = 4;
    opts.min_write_buffer_number_to_merge = 2;

    // --- Compression: low levels uncompressed (write perf), high levels LZ4 (space) ---
    // Note: RocksDB was built with LZ4 only (no ZSTD/Snappy). LZ4 is faster but lower ratio.
    opts.compression_per_level = {
        rocksdb::kNoCompression, rocksdb::kNoCompression, rocksdb::kNoCompression,
        rocksdb::kLZ4Compression, rocksdb::kLZ4Compression, rocksdb::kLZ4Compression,
        rocksdb::kLZ4Compression
    };

    // --- Dynamic level bytes: 90% data sinks to last level, reduces write amplification ---
    opts.level_compaction_dynamic_level_bytes = true;

    // --- BlobDB (KV separation): large values go to blob files, reducing compaction write amp ---
    opts.enable_blob_files     = true;
    opts.min_blob_size         = 512;                  // values > 512B → blob
    opts.blob_compression_type = rocksdb::kLZ4Compression;
    opts.enable_blob_garbage_collection = true;

    // --- Block-based table options: Block Cache + Bloom Filter + cached index/filter ---
    rocksdb::BlockBasedTableOptions table_opts;
    // 512MB LRU block cache (adjust per machine memory)
    table_opts.block_cache = rocksdb::NewLRUCache(512ULL * 1024 * 1024);
    // Full bloom filter (40% faster point lookups vs block-based per RocksDB docs)
    table_opts.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, /*full_filter=*/false));
    // Cache index & filter blocks in block cache, pin L0 to avoid eviction
    table_opts.cache_index_and_filter_blocks = true;
    table_opts.pin_l0_filter_and_index_blocks_in_cache = true;
    // Use 16KB data blocks (default 4KB is too small for our value sizes)
    table_opts.block_size = 16 * 1024;

    opts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opts));
}

// ============================================================
// Constructor / Destructor
// ============================================================

ScalarStorage::ScalarStorage(const std::string& db_path) {
    rocksdb::DBOptions db_opts;
    ConfigureDBOptions(db_opts);

    rocksdb::ColumnFamilyOptions cf_opts;
    ConfigureCFOptions(cf_opts);

    std::vector<rocksdb::ColumnFamilyDescriptor> cf_descs = {
        {rocksdb::kDefaultColumnFamilyName, cf_opts},
        {"scalar", cf_opts},
        {"vector", cf_opts}
    };

    std::vector<rocksdb::ColumnFamilyHandle*> handles;
    rocksdb::Status status = rocksdb::DB::Open(db_opts, db_path, cf_descs, &handles, &db_);

    if (!status.ok()) {
        throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }

    default_cf_handle_ = handles[0];
    scalar_cf_handle_  = handles[1];
    vector_cf_handle_  = handles[2];

    global_logger->info("ScalarStorage opened with 3 CFs (default/scalar/vector), "
                         "bloom filter + LZ4 compression + BlobDB + 512MB block cache");
}

ScalarStorage::~ScalarStorage() {
    delete scalar_cf_handle_;
    delete vector_cf_handle_;
    delete default_cf_handle_;
    delete db_;
}

auto ScalarStorage::MakeKey(const std::string& collection_name, uint64_t id) -> std::string {
    return collection_name + ":" + std::to_string(id);
}

// ============================================================
// Binary TLV Serialization (scalar fields only, skips "vectors")
// ============================================================

auto ScalarStorage::SerializeScalar(const rapidjson::Document& data) -> std::string {
    std::string buf;
    buf.reserve(256);

    PutU8(buf, kScalarMagic);
    PutU8(buf, kScalarVersion);

    // Count scalar fields (exclude "vectors")
    uint16_t num_fields = 0;
    for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
        std::string name = it->name.GetString();
        if (name == "vectors") continue;
        ++num_fields;
    }
    PutU16(buf, num_fields);

    for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
        std::string name = it->name.GetString();
        if (name == "vectors") continue;

        // Field name
        PutU16(buf, static_cast<uint16_t>(name.size()));
        buf.append(name);

        // Type + value
        const rapidjson::Value& v = it->value;
        if (v.IsInt64()) {
            PutU8(buf, kTypeInt64);
            int64_t val = v.GetInt64();
            PutU32(buf, 8);
            PutU64(buf, static_cast<uint64_t>(val));
        } else if (v.IsUint64()) {
            PutU8(buf, kTypeUint64);
            uint64_t val = v.GetUint64();
            PutU32(buf, 8);
            PutU64(buf, val);
        } else if (v.IsInt()) {
            PutU8(buf, kTypeInt64);
            int64_t val = static_cast<int64_t>(v.GetInt());
            PutU32(buf, 8);
            PutU64(buf, static_cast<uint64_t>(val));
        } else if (v.IsUint()) {
            PutU8(buf, kTypeUint64);
            uint64_t val = static_cast<uint64_t>(v.GetUint());
            PutU32(buf, 8);
            PutU64(buf, val);
        } else if (v.IsDouble()) {
            PutU8(buf, kTypeDouble);
            PutU32(buf, 8);
            PutF64(buf, v.GetDouble());
        } else if (v.IsBool()) {
            PutU8(buf, kTypeBool);
            PutU32(buf, 1);
            PutU8(buf, v.GetBool() ? 1 : 0);
        } else if (v.IsString()) {
            PutU8(buf, kTypeString);
            std::string s = v.GetString();
            PutU32(buf, static_cast<uint32_t>(s.size()));
            buf.append(s);
        } else {
            // Unknown type — skip by writing empty
            PutU8(buf, kTypeString);
            PutU32(buf, 0);
        }
    }

    return buf;
}

auto ScalarStorage::DeserializeScalar(const std::string& binary) -> rapidjson::Document {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();

    if (binary.size() < 4) return doc;
    const char* p = binary.data();
    if (static_cast<uint8_t>(p[0]) != kScalarMagic) {
        global_logger->warn("DeserializeScalar: bad magic byte 0x{:02x}", static_cast<uint8_t>(p[0]));
        return doc;
    }
    // p[1] = version (currently 0x01)

    uint16_t num_fields = GetU16(p + 2);
    size_t offset = 4;

    for (uint16_t i = 0; i < num_fields; ++i) {
        if (offset + 2 > binary.size()) break;
        uint16_t name_len = GetU16(p + offset);
        offset += 2;
        if (offset + name_len > binary.size()) break;
        std::string name(p + offset, name_len);
        offset += name_len;

        if (offset + 1 > binary.size()) break;
        uint8_t type = GetU8(p + offset);
        offset += 1;

        if (offset + 4 > binary.size()) break;
        uint32_t val_len = GetU32(p + offset);
        offset += 4;
        if (offset + val_len > binary.size()) break;
        const char* val_ptr = p + offset;

        switch (type) {
            case kTypeInt64: {
                int64_t val = static_cast<int64_t>(GetU64(val_ptr));
                doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(val), alloc);
                break;
            }
            case kTypeUint64: {
                uint64_t val = GetU64(val_ptr);
                doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(val), alloc);
                break;
            }
            case kTypeDouble: {
                double val = GetF64(val_ptr);
                doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(val), alloc);
                break;
            }
            case kTypeBool: {
                bool val = GetU8(val_ptr) != 0;
                doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(val), alloc);
                break;
            }
            case kTypeString: {
                std::string s(val_ptr, val_len);
                doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(s.c_str(), alloc), alloc);
                break;
            }
            default:
                break;
        }

        offset += val_len;
    }

    return doc;
}

// ============================================================
// Vector Serialization (raw float32 LE)
// ============================================================

auto ScalarStorage::SerializeVector(const float* data, size_t count) -> std::string {
    return std::string(reinterpret_cast<const char*>(data), count * sizeof(float));
}

auto ScalarStorage::DeserializeVector(const std::string& binary) -> std::vector<float> {
    if (binary.empty()) return {};
    size_t count = binary.size() / sizeof(float);
    std::vector<float> vec(count);
    std::memcpy(vec.data(), binary.data(), count * sizeof(float));
    return vec;
}

// ============================================================
// Scalar CRUD (scalar CF, binary format)
// ============================================================

void ScalarStorage::InsertScalar(const std::string& collection_name, uint64_t id, const rapidjson::Document& data) {
    std::string value = SerializeScalar(data);
    std::string key = MakeKey(collection_name, id);
    rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), scalar_cf_handle_, key, value);
    if (!status.ok()) {
        global_logger->error("Failed to insert scalar: {}", status.ToString());
    }
}

void ScalarStorage::BatchInsertScalar(const std::string& collection_name, const std::vector<uint64_t>& ids, const std::vector<rapidjson::Document>& datas) {
    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < ids.size(); ++i) {
        std::string value = SerializeScalar(datas[i]);
        batch.Put(scalar_cf_handle_, MakeKey(collection_name, ids[i]), value);
    }
    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
        global_logger->error("Failed to batch insert scalar: {}", status.ToString());
    }
}

auto ScalarStorage::GetScalar(const std::string& collection_name, uint64_t id) -> rapidjson::Document {
    std::string value;
    std::string key = MakeKey(collection_name, id);
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), scalar_cf_handle_, key, &value);
    if (!status.ok()) {
        return rapidjson::Document();  // null document = not found
    }
    return DeserializeScalar(value);
}

// ============================================================
// Vector CRUD (vector CF, raw float32)
// ============================================================

void ScalarStorage::InsertVector(const std::string& collection_name, uint64_t id, const std::vector<float>& vec) {
    std::string value = SerializeVector(vec.data(), vec.size());
    std::string key = MakeKey(collection_name, id);
    rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), vector_cf_handle_, key, value);
    if (!status.ok()) {
        global_logger->error("Failed to insert vector: {}", status.ToString());
    }
}

void ScalarStorage::BatchInsertVector(const std::string& collection_name, const std::vector<uint64_t>& ids, const std::vector<float>& flat_vectors, int dim) {
    if (ids.empty() || dim <= 0) return;
    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < ids.size(); ++i) {
        const float* ptr = flat_vectors.data() + i * dim;
        std::string value = SerializeVector(ptr, static_cast<size_t>(dim));
        batch.Put(vector_cf_handle_, MakeKey(collection_name, ids[i]), value);
    }
    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
        global_logger->error("Failed to batch insert vectors: {}", status.ToString());
    }
}

auto ScalarStorage::GetVector(const std::string& collection_name, uint64_t id) -> std::vector<float> {
    std::string value;
    std::string key = MakeKey(collection_name, id);
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), vector_cf_handle_, key, &value);
    if (!status.ok()) {
        return {};  // empty = not found
    }
    return DeserializeVector(value);
}

}  // namespace vectordb
