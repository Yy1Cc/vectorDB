// GARDEN 连续字段注册 HTTP 入口 + range op 端到端测试
//
// 覆盖：
// 1. 通过 CollectionManager 取 GARDEN_HNSW index 注册连续/离散字段（registerGardenField handler 的底层逻辑）
// 2. vector_database.cpp 新增的 range op 解析：filter {fieldName, op:"range", min, max}
// 3. 单边 >= / <= 条件的 range 默认值修正（避免连续 Pruner 桶收集退化）
// 4. 端到端：注册连续字段 → Upsert GARDEN_HNSW → range 搜索，结果落在区间内

#include "collection/collection_manager.h"
#include "database/vector_database.h"
#include "index/garden_index.h"
#include "index/index_factory.h"
#include <rapidjson/document.h>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class GardenFieldRegisterEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const garden_field_env =
    ::testing::AddGlobalTestEnvironment(new GardenFieldRegisterEnv);

static const int DIM = 8;

static std::vector<float> GenVec(int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(DIM);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 构造带 price（int）的 upsert 文档
static rapidjson::Document MakeUpsertDocWithPrice(const std::vector<float>& vec, int64_t price) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : vec) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("price", price, a);
    return doc;
}

// 构造带 range filter 的 search 文档
static rapidjson::Document MakeRangeSearchDoc(const std::vector<float>& query, int k,
                                              const std::string& field,
                                              int64_t range_min, int64_t range_max) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : query) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", k, a);
    doc.AddMember("indexType", rapidjson::Value("GARDEN_HNSW", a), a);
    rapidjson::Value filter(rapidjson::kObjectType);
    filter.AddMember("fieldName", rapidjson::Value(field.c_str(), a), a);
    filter.AddMember("op", rapidjson::Value("range", a), a);
    filter.AddMember("min", range_min, a);
    filter.AddMember("max", range_max, a);
    doc.AddMember("filter", filter, a);
    return doc;
}

// 构造带 >= 单边 filter 的 search 文档
static rapidjson::Document MakeGeSearchDoc(const std::vector<float>& query, int k,
                                           const std::string& field, int64_t value) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : query) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", k, a);
    doc.AddMember("indexType", rapidjson::Value("GARDEN_HNSW", a), a);
    rapidjson::Value filter(rapidjson::kObjectType);
    filter.AddMember("fieldName", rapidjson::Value(field.c_str(), a), a);
    filter.AddMember("op", rapidjson::Value(">=", a), a);
    filter.AddMember("value", value, a);
    doc.AddMember("filter", filter, a);
    return doc;
}

static const std::string COLL = "garden_field_test";

// 每个测试前确保 collection 干净重建（dim=8），并注册连续字段 price [0,1000] bucket=100
static Collection* EnsureTestCollection() {
    CollectionManager::Instance().DropCollection(COLL);  // 幂等清理
    CollectionManager::Instance().CreateCollection(COLL, DIM, 10000);
    auto* coll = CollectionManager::Instance().GetCollection(COLL);
    EXPECT_NE(coll, nullptr);
    auto* garden = static_cast<GardenIndex*>(coll->GetIndex(IndexFactory::IndexType::GARDEN_HNSW));
    EXPECT_NE(garden, nullptr);
    garden->RegisterContinuousField("price", 0, 1000, 100);
    return coll;
}

// ========== 测试1：通过 CollectionManager 注册连续字段，子图就位 ==========
TEST(GardenFieldRegisterTest, RegisterContinuousFieldViaManager) {
    auto* coll = EnsureTestCollection();
    auto* garden = static_cast<GardenIndex*>(coll->GetIndex(IndexFactory::IndexType::GARDEN_HNSW));
    EXPECT_TRUE(garden->HasContinuousSubgraph("price"));
    EXPECT_FALSE(garden->HasContinuousSubgraph("nonexistent"));
}

// ========== 测试2：通过 CollectionManager 注册离散字段 ==========
TEST(GardenFieldRegisterTest, RegisterDiscreteFieldViaManager) {
    CollectionManager::Instance().DropCollection(COLL);
    CollectionManager::Instance().CreateCollection(COLL, DIM, 10000);
    auto* coll = CollectionManager::Instance().GetCollection(COLL);
    auto* garden = static_cast<GardenIndex*>(coll->GetIndex(IndexFactory::IndexType::GARDEN_HNSW));
    garden->RegisterDiscreteField("brand");
    EXPECT_TRUE(garden->HasDiscreteSubgraph("brand"));
}

// ========== 测试3：range op 端到端，结果落在区间内 ==========
TEST(GardenFieldRegisterTest, RangeSearchViaDatabase) {
    system("rm -rf /tmp/vdb_garden_range && mkdir -p /tmp/vdb_garden_range");
    VectorDatabase db("/tmp/vdb_garden_range/storage", "/tmp/vdb_garden_range/wal.log");
    EnsureTestCollection();

    // price 0-90 的向量在 0.1 附近（id 0-9），price 200-290 在 0.9 附近（id 100-109）
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.9f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(100 + i), MakeUpsertDocWithPrice(vec, 200 + i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }

    // 搜索 price 0-99 范围，query 在 0.1 区域
    std::vector<float> query(DIM, 0.1f);
    auto [ids, dists] = db.Search(COLL, MakeRangeSearchDoc(query, 5, "price", 0, 99));

    int valid = 0;
    for (int64_t id : ids) {
        if (id >= 0) {
            valid++;
            EXPECT_LT(id, 10) << "id " << id << " should be in price 0-99 (id 0-9)";
        }
    }
    EXPECT_GE(valid, 1) << "should return at least one result in range";
}

// ========== 测试4：range op 排除区间外的向量 ==========
TEST(GardenFieldRegisterTest, RangeSearchExcludesOutOfRange) {
    system("rm -rf /tmp/vdb_garden_range2 && mkdir -p /tmp/vdb_garden_range2");
    VectorDatabase db("/tmp/vdb_garden_range2/storage", "/tmp/vdb_garden_range2/wal.log");
    EnsureTestCollection();

    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);  // 同区域，但 price 高
        db.Upsert(COLL, static_cast<uint64_t>(100 + i), MakeUpsertDocWithPrice(vec, 500 + i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }

    // range 0-99 应只返回 id 0-9，排除 id 100+（price 500+）
    std::vector<float> query(DIM, 0.1f);
    auto [ids, dists] = db.Search(COLL, MakeRangeSearchDoc(query, 10, "price", 0, 99));

    for (int64_t id : ids) {
        if (id >= 0) {
            EXPECT_LT(id, 100) << "id " << id << " (price>=500) should be excluded by range 0-99";
        }
    }
}

// ========== 测试5：单边 >= 条件不退化（修正 range_max 默认值） ==========
// 修复前：>= 只设 range_min，range_max=0 导致 GARDEN 桶收集退化
// 修复后：>= 设 range_max=INT64_MAX，桶收集正常
TEST(GardenFieldRegisterTest, GeFilterDoesNotDegrade) {
    system("rm -rf /tmp/vdb_garden_ge && mkdir -p /tmp/vdb_garden_ge");
    VectorDatabase db("/tmp/vdb_garden_ge/storage", "/tmp/vdb_garden_ge/wal.log");
    EnsureTestCollection();

    // id 0-4: price 0-40（低），id 100-104: price 500-540（高），向量同区域
    for (int i = 0; i < 5; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }
    for (int i = 0; i < 5; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(100 + i), MakeUpsertDocWithPrice(vec, 500 + i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }

    // price >= 200 应返回 id 100-104，排除 id 0-4（price < 200）
    std::vector<float> query(DIM, 0.1f);
    auto [ids, dists] = db.Search(COLL, MakeGeSearchDoc(query, 5, "price", 200));

    bool found_high = false;
    for (int64_t id : ids) {
        if (id >= 0) {
            EXPECT_GE(id, 100) << "id " << id << " (price<200) should be excluded by >=200";
            if (id >= 100) found_high = true;
        }
    }
    EXPECT_TRUE(found_high) << "should return at least one id with price>=200";
}

// 构造带 excludeIds 的 GARDEN 搜索文档（无 filter）
static rapidjson::Document MakeExcludeSearchDoc(const std::vector<float>& query, int k,
                                                const std::vector<uint64_t>& exclude_ids) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : query) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", k, a);
    doc.AddMember("indexType", rapidjson::Value("GARDEN_HNSW", a), a);
    rapidjson::Value excl(rapidjson::kArrayType);
    for (uint64_t id : exclude_ids) excl.PushBack(id, a);
    doc.AddMember("excludeIds", excl, a);
    return doc;
}

// 构造 range filter + excludeIds 的 GARDEN 搜索文档
static rapidjson::Document MakeRangeExcludeSearchDoc(const std::vector<float>& query, int k,
                                                     const std::string& field,
                                                     int64_t rmin, int64_t rmax,
                                                     const std::vector<uint64_t>& exclude_ids) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    for (float v : query) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", k, a);
    doc.AddMember("indexType", rapidjson::Value("GARDEN_HNSW", a), a);
    rapidjson::Value filter(rapidjson::kObjectType);
    filter.AddMember("fieldName", rapidjson::Value(field.c_str(), a), a);
    filter.AddMember("op", rapidjson::Value("range", a), a);
    filter.AddMember("min", rmin, a);
    filter.AddMember("max", rmax, a);
    doc.AddMember("filter", filter, a);
    rapidjson::Value excl(rapidjson::kArrayType);
    for (uint64_t id : exclude_ids) excl.PushBack(id, a);
    doc.AddMember("excludeIds", excl, a);
    return doc;
}

// ========== 测试6：GARDEN excludeIds 黑名单（无 filter，走暴力兜底） ==========
TEST(GardenFieldRegisterTest, ExcludeIdsGardenNoFilter) {
    system("rm -rf /tmp/vdb_garden_excl1 && mkdir -p /tmp/vdb_garden_excl1");
    VectorDatabase db("/tmp/vdb_garden_excl1/storage", "/tmp/vdb_garden_excl1/wal.log");
    EnsureTestCollection();

    // id 0-9 向量都在 0.1 附近
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }

    std::vector<float> query(DIM, 0.1f);
    auto [ids, dists] = db.Search(COLL, MakeExcludeSearchDoc(query, 10, {0, 1, 2}));

    for (int64_t id : ids) {
        if (id >= 0) {
            EXPECT_TRUE(id != 0 && id != 1 && id != 2)
                << "id " << id << " should be excluded by excludeIds";
        }
    }
    bool found = false;
    for (int64_t id : ids) if (id >= 0) found = true;
    EXPECT_TRUE(found) << "should return at least one non-excluded id";
}

// ========== 测试7：GARDEN excludeIds + range filter 组合 ==========
TEST(GardenFieldRegisterTest, ExcludeIdsGardenWithRange) {
    system("rm -rf /tmp/vdb_garden_excl2 && mkdir -p /tmp/vdb_garden_excl2");
    VectorDatabase db("/tmp/vdb_garden_excl2/storage", "/tmp/vdb_garden_excl2/wal.log");
    EnsureTestCollection();

    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::GARDEN_HNSW);
    }

    // range price 0-99（id 0-9）但排除 0,1
    std::vector<float> query(DIM, 0.1f);
    auto [ids, dists] = db.Search(COLL, MakeRangeExcludeSearchDoc(query, 10, "price", 0, 99, {0, 1}));

    for (int64_t id : ids) {
        if (id >= 0) {
            EXPECT_TRUE(id != 0 && id != 1) << "id " << id << " should be excluded";
            EXPECT_LT(id, 10) << "id " << id << " should be in range 0-99";
        }
    }
}

// ========== 测试8：通用路径（FLAT）excludeIds 后过滤 ==========
TEST(GardenFieldRegisterTest, ExcludeIdsGenericPath) {
    system("rm -rf /tmp/vdb_garden_excl3 && mkdir -p /tmp/vdb_garden_excl3");
    VectorDatabase db("/tmp/vdb_garden_excl3/storage", "/tmp/vdb_garden_excl3/wal.log");
    EnsureTestCollection();

    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 0.1f + i * 0.001f);
        db.Upsert(COLL, static_cast<uint64_t>(i), MakeUpsertDocWithPrice(vec, i * 10),
                  IndexFactory::IndexType::FLAT);
    }

    rapidjson::Document doc;
    doc.SetObject();
    auto& a = doc.GetAllocator();
    rapidjson::Value vectors(rapidjson::kArrayType);
    std::vector<float> q(DIM, 0.1f);
    for (float v : q) vectors.PushBack(v, a);
    doc.AddMember("vectors", vectors, a);
    doc.AddMember("k", 10, a);
    doc.AddMember("indexType", rapidjson::Value("FLAT", a), a);
    rapidjson::Value excl(rapidjson::kArrayType);
    excl.PushBack(static_cast<uint64_t>(0), a);
    excl.PushBack(static_cast<uint64_t>(1), a);
    excl.PushBack(static_cast<uint64_t>(2), a);
    doc.AddMember("excludeIds", excl, a);

    auto [ids, dists] = db.Search(COLL, doc);

    for (int64_t id : ids) {
        if (id >= 0) {
            EXPECT_TRUE(id != 0 && id != 1 && id != 2)
                << "id " << id << " should be excluded by excludeIds (generic path)";
        }
    }
}

}  // namespace vectordb
