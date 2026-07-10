#include "database/vector_database.h"
#include "index/index_factory.h"
#include <rapidjson/document.h>
#include <logger/logger.h>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class E2EEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const e2e_env =
    ::testing::AddGlobalTestEnvironment(new E2EEnvironment);

static const int DIM = 128;

static std::vector<float> GenVec(int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(DIM);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 构造 Upsert 数据文档
static rapidjson::Document MakeUpsertDoc(const std::vector<float>& vec,
                                          const std::map<std::string, std::string>& strs = {},
                                          int64_t ttl = -1) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : vec) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    for (const auto& [k, v] : strs) {
        rapidjson::Value key(k.c_str(), a);
        rapidjson::Value val(v.c_str(), a);
        doc.AddMember(key, val, a);
    }
    if (ttl > 0) doc.AddMember("ttl", ttl, a);
    return doc;
}

// 构造 Search 请求文档
static rapidjson::Document MakeSearchDoc(const std::vector<float>& query, int k,
                                          const std::string& idx_type = "FLAT") {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : query) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", k, a);
    doc.AddMember("indexType", rapidjson::Value(idx_type.c_str(), a), a);
    return doc;
}

// ========== FullText 全文搜索端到端 ==========

TEST(E2EIntegrationTest, FullTextSearchViaDatabase) {
    // 清理旧数据并创建目录
    system("rm -rf /tmp/vdb_e2e_ft && mkdir -p /tmp/vdb_e2e_ft");
    VectorDatabase db("/tmp/vdb_e2e_ft/storage", "/tmp/vdb_e2e_ft/wal.log");

    auto v1 = GenVec(1);
    auto v2 = GenVec(2);

    db.Upsert(1, MakeUpsertDoc(v1, {{"content", "vector database search engine"}}),
              IndexFactory::IndexType::FLAT);
    db.Upsert(2, MakeUpsertDoc(v2, {{"content", "machine learning models"}}),
              IndexFactory::IndexType::FLAT);

    // 全文搜索 "vector"，应只返回 id=1
    auto search_doc = MakeSearchDoc(v1, 10, "FLAT");
    auto& a = search_doc.GetAllocator();
    rapidjson::Value ft(rapidjson::kObjectType);
    ft.AddMember("field", "content", a);
    ft.AddMember("query", "vector", a);
    search_doc.AddMember("fulltext", ft, a);

    auto [ids, dists] = db.Search(search_doc);

    // id=1 包含 "vector"，应该返回；id=2 不包含
    bool found1 = false;
    bool found2 = false;
    for (int64_t id : ids) {
        if (id == 1) found1 = true;
        if (id == 2) found2 = true;
    }
    EXPECT_TRUE(found1) << "id=1 should be found (contains 'vector')";
    EXPECT_FALSE(found2) << "id=2 should not be found (no 'vector')";
}

// ========== TTL 过期过滤端到端 ==========

TEST(E2EIntegrationTest, TTLFilterViaDatabase) {
    system("rm -rf /tmp/vdb_e2e_ttl && mkdir -p /tmp/vdb_e2e_ttl");
    VectorDatabase db("/tmp/vdb_e2e_ttl/storage", "/tmp/vdb_e2e_ttl/wal.log");

    auto v1 = GenVec(10);
    auto v2 = GenVec(20);

    // id=100: TTL=1秒（马上过期）
    db.Upsert(100, MakeUpsertDoc(v1, {}, 1), IndexFactory::IndexType::FLAT);
    // id=200: TTL=3600秒
    db.Upsert(200, MakeUpsertDoc(v2, {}, 3600), IndexFactory::IndexType::FLAT);

    // 立即搜索，两个都应该返回
    auto [ids1, _1] = db.Search(MakeSearchDoc(v1, 10, "FLAT"));
    bool found100_before = false, found200_before = false;
    for (int64_t id : ids1) {
        if (id == 100) found100_before = true;
        if (id == 200) found200_before = true;
    }
    EXPECT_TRUE(found100_before) << "id=100 should be found before expiry";
    EXPECT_TRUE(found200_before) << "id=200 should be found before expiry";

    // 等待2秒让 id=100 过期
    std::this_thread::sleep_for(std::chrono::seconds(2));

    auto [ids2, _2] = db.Search(MakeSearchDoc(v1, 10, "FLAT"));
    bool found100_after = false, found200_after = false;
    for (int64_t id : ids2) {
        if (id == 100) found100_after = true;
        if (id == 200) found200_after = true;
    }
    EXPECT_FALSE(found100_after) << "id=100 should be expired and filtered out";
    EXPECT_TRUE(found200_after) << "id=200 should still be found";

    // CleanExpiredVectors 清理 TTL 记录
    db.CleanExpiredVectors();
}

// ========== LAYERED_FLAT 分层索引端到端 ==========

TEST(E2EIntegrationTest, LayeredFlatViaDatabase) {
    system("rm -rf /tmp/vdb_e2e_layered && mkdir -p /tmp/vdb_e2e_layered");
    // 手动初始化 LAYERED_FLAT 索引（VdbServerInit 没有初始化这个类型）
    auto& factory = IndexFactory::Instance();
    factory.Init(IndexFactory::IndexType::LAYERED_FLAT, DIM, 1000000);

    VectorDatabase db("/tmp/vdb_e2e_layered/storage", "/tmp/vdb_e2e_layered/wal.log");

    // 插入 5 个向量到 LAYERED_FLAT
    for (int i = 1; i <= 5; ++i) {
        auto v = GenVec(i * 100);
        db.Upsert(i, MakeUpsertDoc(v), IndexFactory::IndexType::LAYERED_FLAT);
    }

    // 搜索自身
    auto v1 = GenVec(100);
    auto [ids, dists] = db.Search(MakeSearchDoc(v1, 3, "LAYERED_FLAT"));

    ASSERT_GE(ids.size(), 1u);
    EXPECT_EQ(ids[0], 1) << "top-1 should be id=1 (self)";
}

// ========== LAYERED_FLAT + FullText 组合 ==========

TEST(E2EIntegrationTest, LayeredFlatWithFullText) {
    system("rm -rf /tmp/vdb_e2e_lft && mkdir -p /tmp/vdb_e2e_lft");
    auto& factory = IndexFactory::Instance();
    factory.Init(IndexFactory::IndexType::LAYERED_FLAT, DIM, 1000000);

    VectorDatabase db("/tmp/vdb_e2e_lft/storage", "/tmp/vdb_e2e_lft/wal.log");

    db.Upsert(1, MakeUpsertDoc(GenVec(1), {{"title", "vector search system"}}),
              IndexFactory::IndexType::LAYERED_FLAT);
    db.Upsert(2, MakeUpsertDoc(GenVec(2), {{"title", "image recognition model"}}),
              IndexFactory::IndexType::LAYERED_FLAT);

    // 全文搜索 "search" + 向量搜索
    auto search_doc = MakeSearchDoc(GenVec(1), 10, "LAYERED_FLAT");
    auto& a = search_doc.GetAllocator();
    rapidjson::Value ft(rapidjson::kObjectType);
    ft.AddMember("field", "title", a);
    ft.AddMember("query", "search", a);
    search_doc.AddMember("fulltext", ft, a);

    auto [ids, _] = db.Search(search_doc);

    bool found1 = false;
    for (int64_t id : ids) {
        if (id == 1) found1 = true;
    }
    EXPECT_TRUE(found1) << "id=1 should be found (title contains 'search')";
}

}  // namespace vectordb
