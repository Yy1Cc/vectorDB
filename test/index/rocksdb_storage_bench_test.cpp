// RocksDB 存储层对比压测：为面试提供量化数据
// 对比维度：
//   1. TLV 二进制序列化 vs JSON 序列化（size + 序列化/反序列化延迟）
//   2. BlobDB on/off（写入吞吐 + 写放大，证明 KV 分离对大 value 的收益）
//   3. 压缩 LZ4 vs None（空间占用对比）
//
// 运行：./test/rocksdb_storage_bench_test
// 直接看 stdout 的对比表格。不依赖 vectorDB 全局状态，只用 RocksDB + rapidjson。

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/cache.h>
#include <rocksdb/statistics.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "gtest/gtest.h"

namespace {

// ============================================================
// TLV 二进制编码（与 src/database/scalar_storage.cpp 完全一致）
// ============================================================
constexpr uint8_t kScalarMagic   = 0xDB;
constexpr uint8_t kScalarVersion = 0x01;
constexpr uint8_t kTypeInt64     = 0;
constexpr uint8_t kTypeUint64    = 1;
constexpr uint8_t kTypeString    = 2;
constexpr uint8_t kTypeDouble    = 3;
constexpr uint8_t kTypeBool      = 4;

inline void PutU8 (std::string& buf, uint8_t  v) { buf.push_back(static_cast<char>(v)); }
inline void PutU16(std::string& buf, uint16_t v) { buf.append(reinterpret_cast<const char*>(&v), 2); }
inline void PutU32(std::string& buf, uint32_t v) { buf.append(reinterpret_cast<const char*>(&v), 4); }
inline void PutU64(std::string& buf, uint64_t v) { buf.append(reinterpret_cast<const char*>(&v), 8); }
inline uint16_t GetU16(const char* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t GetU32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t GetU64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

std::string SerializeTLV(const rapidjson::Document& data) {
    std::string buf;
    buf.reserve(256);
    PutU8(buf, kScalarMagic);
    PutU8(buf, kScalarVersion);
    uint16_t num_fields = 0;
    for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
        if (std::string(it->name.GetString()) == "vectors") continue;
        ++num_fields;
    }
    PutU16(buf, num_fields);
    for (auto it = data.MemberBegin(); it != data.MemberEnd(); ++it) {
        std::string name = it->name.GetString();
        if (name == "vectors") continue;
        PutU16(buf, static_cast<uint16_t>(name.size()));
        buf.append(name);
        const rapidjson::Value& v = it->value;
        if (v.IsInt64()) {
            PutU8(buf, kTypeInt64); PutU32(buf, 8);
            PutU64(buf, static_cast<uint64_t>(v.GetInt64()));
        } else if (v.IsUint64()) {
            PutU8(buf, kTypeUint64); PutU32(buf, 8);
            PutU64(buf, v.GetUint64());
        } else if (v.IsInt()) {
            PutU8(buf, kTypeInt64); PutU32(buf, 8);
            PutU64(buf, static_cast<uint64_t>(v.GetInt()));
        } else if (v.IsUint()) {
            PutU8(buf, kTypeUint64); PutU32(buf, 8);
            PutU64(buf, static_cast<uint64_t>(v.GetUint()));
        } else if (v.IsDouble()) {
            PutU8(buf, kTypeDouble); PutU32(buf, 8);
            buf.append(reinterpret_cast<const char*>(&v), 8);
        } else if (v.IsBool()) {
            PutU8(buf, kTypeBool); PutU32(buf, 1);
            PutU8(buf, v.GetBool() ? 1 : 0);
        } else if (v.IsString()) {
            PutU8(buf, kTypeString);
            std::string s = v.GetString();
            PutU32(buf, static_cast<uint32_t>(s.size()));
            buf.append(s);
        } else {
            PutU8(buf, kTypeString); PutU32(buf, 0);
        }
    }
    return buf;
}

rapidjson::Document DeserializeTLV(const std::string& binary) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();
    if (binary.size() < 4) return doc;
    const char* p = binary.data();
    if (static_cast<uint8_t>(p[0]) != kScalarMagic) return doc;
    uint16_t num_fields = GetU16(p + 2);
    size_t offset = 4;
    for (uint16_t i = 0; i < num_fields; ++i) {
        if (offset + 2 > binary.size()) break;
        uint16_t name_len = GetU16(p + offset); offset += 2;
        if (offset + name_len > binary.size()) break;
        std::string name(p + offset, name_len); offset += name_len;
        if (offset + 1 > binary.size()) break;
        uint8_t type = static_cast<uint8_t>(p[offset]); offset += 1;
        if (offset + 4 > binary.size()) break;
        uint32_t val_len = GetU32(p + offset); offset += 4;
        if (offset + val_len > binary.size()) break;
        const char* val_ptr = p + offset;
        switch (type) {
            case kTypeInt64:  doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(static_cast<int64_t>(GetU64(val_ptr))), alloc); break;
            case kTypeUint64: doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(GetU64(val_ptr)), alloc); break;
            case kTypeDouble: { double d; std::memcpy(&d, val_ptr, 8); doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(d), alloc); break; }
            case kTypeBool:   doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(GetU16(val_ptr) != 0), alloc); break;
            case kTypeString: doc.AddMember(rapidjson::Value(name.c_str(), alloc), rapidjson::Value(std::string(val_ptr, val_len).c_str(), alloc), alloc); break;
            default: break;
        }
        offset += val_len;
    }
    return doc;
}

std::string SerializeJSON(const rapidjson::Document& data) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    data.Accept(w);
    return buf.GetString();
}

// 构造典型标量文档（模拟向量库的标量字段：id + int 标签 + 字符串标签 + price）
rapidjson::Document MakeScalarDoc(int id) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    doc.AddMember("id", id, a);
    doc.AddMember("int_field", id * 7 % 1000, a);
    doc.AddMember("label", rapidjson::Value(("tag_" + std::to_string(id % 20)).c_str(), a), a);
    doc.AddMember("price", static_cast<int64_t>(id * 13 % 100000), a);
    doc.AddMember("active", (id % 3 == 0), a);
    doc.AddMember("score", static_cast<double>(id) * 0.5, a);
    return doc;
}

// 计时辅助
using Clock = std::chrono::high_resolution_clock;
template <typename Fn>
double MeasureMs(int repeat, Fn&& fn) {
    auto t0 = Clock::now();
    for (int i = 0; i < repeat; ++i) fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void RmDir(const std::string& path) {
    std::string cmd = "rm -rf " + path;
    std::system(cmd.c_str());
}

// 配置一个 RocksDB 实例（可控制 BlobDB / 压缩）
rocksdb::DB* OpenDB(const std::string& path, bool blob_on, bool compress,
                    size_t memtable_size = 64ULL * 1024 * 1024,
                    bool dynamic_level = true) {
    rocksdb::DBOptions db_opts;
    db_opts.create_if_missing = true;
    db_opts.create_missing_column_families = true;
    db_opts.max_background_jobs = 8;
    db_opts.statistics = rocksdb::CreateDBStatistics();  // 开启统计以读 BYTES_WRITTEN ticker

    rocksdb::ColumnFamilyOptions cf_opts;
    cf_opts.write_buffer_size = memtable_size;  // memtable 大小（小值触发频繁 flush 加速 compaction）
    cf_opts.level_compaction_dynamic_level_bytes = dynamic_level;

    if (blob_on) {
        cf_opts.enable_blob_files = true;
        cf_opts.min_blob_size = 512;
        cf_opts.blob_compression_type = rocksdb::kLZ4Compression;
        cf_opts.enable_blob_garbage_collection = true;
    }
    if (compress) {
        cf_opts.compression_per_level = {
            rocksdb::kNoCompression, rocksdb::kNoCompression, rocksdb::kNoCompression,
            rocksdb::kLZ4Compression, rocksdb::kLZ4Compression, rocksdb::kLZ4Compression,
            rocksdb::kLZ4Compression};
    } else {
        cf_opts.compression_per_level = {rocksdb::kNoCompression, rocksdb::kNoCompression,
                                         rocksdb::kNoCompression, rocksdb::kNoCompression,
                                         rocksdb::kNoCompression, rocksdb::kNoCompression,
                                         rocksdb::kNoCompression};
    }

    rocksdb::BlockBasedTableOptions t;
    t.block_cache = rocksdb::NewLRUCache(128ULL * 1024 * 1024);
    t.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    t.cache_index_and_filter_blocks = true;
    cf_opts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(t));

    rocksdb::DB* db = nullptr;
    rocksdb::Options opts(db_opts, cf_opts);
    rocksdb::Status s = rocksdb::DB::Open(opts, path, &db);
    if (!s.ok()) {
        std::fprintf(stderr, "OpenDB failed: %s\n", s.ToString().c_str());
        return nullptr;
    }
    return db;
}

uint64_t GetBytesWritten(rocksdb::DB* db) {
    auto stat = db->GetDBOptions().statistics;
    if (!stat) return 0;
    // LSM 树总写入（flush + compaction 重写），不含 user-put 和 blob 写入
    // BlobDB on 时 LSM 只含小索引，此值远小于 off（off 重写大 value）
    return stat->getTickerCount(rocksdb::FLUSH_WRITE_BYTES)
         + stat->getTickerCount(rocksdb::COMPACT_WRITE_BYTES);
}

uint64_t GetLiveDataSize(rocksdb::DB* db) {
    uint64_t v = 0;
    db->GetIntProperty("rocksdb.estimate-live-data-size", &v);
    return v;
}

}  // namespace

// ============================================================
// TEST 1: TLV vs JSON 序列化对比
// ============================================================
TEST(RocksDBStorageBench, TLVvsJSON) {
    const int N = 10000;
    std::vector<rapidjson::Document> docs;
    docs.reserve(N);
    for (int i = 0; i < N; ++i) docs.push_back(MakeScalarDoc(i));

    // size 对比
    size_t tlv_total = 0, json_total = 0;
    for (auto& d : docs) {
        tlv_total += SerializeTLV(d).size();
        json_total += SerializeJSON(d).size();
    }

    // 序列化耗时
    double tlv_ser_ms = MeasureMs(N, [&]() { SerializeTLV(docs[N - 1]); });
    double json_ser_ms = MeasureMs(N, [&]() { SerializeJSON(docs[N - 1]); });

    // 反序列化耗时
    std::string tlv_bin = SerializeTLV(docs[0]);
    std::string json_bin = SerializeJSON(docs[0]);
    double tlv_deser_ms = MeasureMs(N, [&]() { DeserializeTLV(tlv_bin); });
    double json_deser_ms = MeasureMs(N, [&]() {
        rapidjson::Document d; d.Parse(json_bin.c_str());
    });

    double tlv_avg = static_cast<double>(tlv_total) / N;
    double json_avg = static_cast<double>(json_total) / N;

    std::printf("\n========== TLV vs JSON 序列化对比 (N=%d) ==========\n", N);
    std::printf("| 指标            | TLV        | JSON       | TLV/JSON |\n");
    std::printf("|-----------------|------------|------------|----------|\n");
    std::printf("| 平均 size (B)   | %-10.1f | %-10.1f | %.2fx    |\n", tlv_avg, json_avg, tlv_avg / json_avg);
    std::printf("| 总 size (KB)    | %-10.1f | %-10.1f | %.2fx    |\n", tlv_total / 1024.0, json_total / 1024.0, static_cast<double>(tlv_total) / json_total);
    std::printf("| 序列化 ops/ms   | %-10.1f | %-10.1f | %.2fx    |\n", N / tlv_ser_ms, N / json_ser_ms, (N / tlv_ser_ms) / (N / json_ser_ms));
    std::printf("| 反序列化 ops/ms | %-10.1f | %-10.1f | %.2fx    |\n", N / tlv_deser_ms, N / json_deser_ms, (N / tlv_deser_ms) / (N / json_deser_ms));
    std::printf("=====================================================\n");
}

// ============================================================
// TEST 2: BlobDB on/off 写入吞吐 + 写放大
// ============================================================
TEST(RocksDBStorageBench, BlobDBOnOff) {
    const int N = 20000;
    const int DIM = 128;
    const size_t vec_bytes = DIM * sizeof(float);  // 512 字节

    // 生成向量数据
    std::vector<std::vector<float>> vecs(N, std::vector<float>(DIM, 0.0f));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j) vecs[i][j] = static_cast<float>(i + j) * 0.001f;

    uint64_t user_payload = static_cast<uint64_t>(N) * (16 + vec_bytes);  // key(16) + value(512)

    auto run = [&](bool blob_on, const std::string& tag) -> void {
        std::string path = "/tmp/rdb_bench_blob_" + std::to_string(blob_on);
        RmDir(path);
        rocksdb::DB* db = OpenDB(path, blob_on, /*compress=*/true);
        ASSERT_NE(db, nullptr);

        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            std::string key = "vec:" + std::to_string(i);
            rocksdb::Slice val(reinterpret_cast<const char*>(vecs[i].data()), vec_bytes);
            db->Put(rocksdb::WriteOptions(), key, val);
        }
        db->Flush(rocksdb::FlushOptions());
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);  // 触发 compaction 让写放大稳定
        uint64_t bytes_written = GetBytesWritten(db);
        uint64_t live_size = GetLiveDataSize(db);
        double qps = N / (ms / 1000.0);
        double write_amp = static_cast<double>(bytes_written) / user_payload;
        double space_amp = static_cast<double>(live_size) / user_payload;

        std::printf("| %-14s | %-10.0f | %-10.1f | %-8.2f | %-8.2f | %-10.2f |\n",
                    tag.c_str(), bytes_written / 1024.0 / 1024.0, qps, write_amp, space_amp, ms / 1000.0);

        delete db;
        RmDir(path);
    };

    std::printf("\n========== BlobDB on/off 对比 (N=%d, vec=%dx%d=%zuB) ==========\n", N, DIM, DIM, vec_bytes);
    std::printf("user payload = %.2f MB\n", user_payload / 1024.0 / 1024.0);
    std::printf("| 配置           | 写入(MB)   | QPS        | 写放大   | 空间放大 | 耗时(s)    |\n");
    std::printf("|----------------|------------|------------|----------|----------|------------|\n");
    run(false, "BlobDB off");
    run(true, "BlobDB on");
    std::printf("=================================================================\n");
    std::printf("注：写放大 = rocksdb.bytes-written / user_payload（含 WAL+flush+compaction+blob 重写）\n");
    std::printf("    空间放大 = live-data-size / user_payload\n");
}

// ============================================================
// TEST 3: 压缩 LZ4 vs None 空间对比
// ============================================================
TEST(RocksDBStorageBench, CompressionOnOff) {
    const int N = 20000;
    const int DIM = 128;
    const size_t vec_bytes = DIM * sizeof(float);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(DIM, 0.0f));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j) vecs[i][j] = static_cast<float>(i + j) * 0.001f;
    uint64_t user_payload = static_cast<uint64_t>(N) * (16 + vec_bytes);

    auto run = [&](bool compress, const std::string& tag) -> void {
        std::string path = "/tmp/rdb_bench_comp_" + std::to_string(compress);
        RmDir(path);
        rocksdb::DB* db = OpenDB(path, /*blob=*/false, compress);
        ASSERT_NE(db, nullptr);
        for (int i = 0; i < N; ++i) {
            std::string key = "vec:" + std::to_string(i);
            rocksdb::Slice val(reinterpret_cast<const char*>(vecs[i].data()), vec_bytes);
            db->Put(rocksdb::WriteOptions(), key, val);
        }
        db->Flush(rocksdb::FlushOptions());
        db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
        uint64_t live_size = GetLiveDataSize(db);
        std::printf("| %-16s | %-12.2f | %-10.2f |\n",
                    tag.c_str(), live_size / 1024.0 / 1024.0,
                    static_cast<double>(live_size) / user_payload);
        delete db;
        RmDir(path);
    };

    std::printf("\n========== 压缩 LZ4 vs None 空间对比 (N=%d) ==========\n", N);
    std::printf("user payload = %.2f MB\n", user_payload / 1024.0 / 1024.0);
    std::printf("| 配置             | 磁盘占用(MB) | 空间比    |\n");
    std::printf("|------------------|--------------|-----------|\n");
    run(false, "NoCompression");
    run(true, "LZ4 (per-level)");
    std::printf("=======================================================\n");
}

// ============================================================
// TEST 4: 大数据集 BlobDB on/off 对比（1M 条，触发多轮 compaction）
// 小 memtable(16MB) + 512MB 数据 → 多轮 flush/compaction → 写放大>1，BlobDB 收益显现
// ============================================================
TEST(RocksDBStorageBench, BlobDBLargeScale) {
    const int N = 1000000;  // 1M 条（百万级）
    const int DIM = 128;
    const size_t vec_bytes = DIM * sizeof(float);  // 512B
    const uint64_t user_payload = static_cast<uint64_t>(N) * (16 + vec_bytes);  // ~521MB

    std::printf("\n========== 大数据集 BlobDB on/off 对比 (N=%d, vec=%dx%d=%zuB, payload=%.1fMB) ==========\n",
                N, DIM, DIM, vec_bytes, user_payload / 1024.0 / 1024.0);
    std::printf("memtable=16MB（强制频繁 flush 触发多轮 compaction）\n");
    std::printf("| 配置           | 写入(MB)   | QPS        | 写放大   | 空间放大 | 耗时(s)   |\n");
    std::printf("|----------------|------------|------------|----------|----------|-----------|\n");

    auto run = [&](bool blob_on, const std::string& tag) -> void {
        std::string path = "/tmp/rdb_bench_large_" + std::to_string(blob_on);
        RmDir(path);
        rocksdb::DB* db = OpenDB(path, blob_on, /*compress=*/true, /*memtable_size=*/16ULL * 1024 * 1024,
                                 /*dynamic_level=*/false);  // 关闭 dynamic level，用经典 leveled 触发逐级写放大
        ASSERT_NE(db, nullptr);

        std::vector<float> vec(DIM);
        auto t0 = Clock::now();
        // 批量写入（每 1000 条一个 WriteBatch），加速写入
        for (int i = 0; i < N; i += 1000) {
            rocksdb::WriteBatch batch;
            int batch_n = std::min(1000, N - i);
            for (int j = 0; j < batch_n; ++j) {
                for (int d = 0; d < DIM; ++d) vec[d] = static_cast<float>((i + j) * 0.001 + d * 0.0001);
                std::string key = "vec:" + std::to_string(i + j);
                batch.Put(key, rocksdb::Slice(reinterpret_cast<const char*>(vec.data()), vec_bytes));
            }
            db->Write(rocksdb::WriteOptions(), &batch);
        }
        db->Flush(rocksdb::FlushOptions());
        // 不手动 CompactRange，等待后台自动多级 compaction（L0→L1→L2...）完成
        uint64_t pending = 1;
        for (int wait = 0; wait < 60 && pending > 0; ++wait) {
            db->GetIntProperty("rocksdb.compaction-pending", &pending);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        uint64_t bytes_written = GetBytesWritten(db);
        uint64_t live = GetLiveDataSize(db);
        double qps = N / (ms / 1000.0);
        double write_amp = static_cast<double>(bytes_written) / user_payload;
        double space_amp = static_cast<double>(live) / user_payload;

        std::printf("| %-14s | %-10.1f | %-10.0f | %-8.2f | %-8.2f | %-9.1f |\n",
                    tag.c_str(), bytes_written / 1024.0 / 1024.0, qps, write_amp, space_amp, ms / 1000.0);

        delete db;
        RmDir(path);
    };

    run(false, "BlobDB off");
    run(true, "BlobDB on");
    std::printf("==================================================================================\n");
    std::printf("注：写放大 = BYTES_WRITTEN ticker / user_payload（含 WAL+flush+compaction+blob 重写）\n");
    std::printf("    大数据集 + 小 memtable 触发多轮 compaction，写放大>1，BlobDB 减少大 value 重写的收益应显现\n");
}
