#include "database/vector_database.h"
#include "collection/collection_manager.h"
#include "index/index_factory.h"
#include <rapidjson/document.h>
#include <logger/logger.h>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class MultiCollectionEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const multi_coll_env =
    ::testing::AddGlobalTestEnvironment(new MultiCollectionEnvironment);

// 生成指定维度的随机向量
static std::vector<float> GenVec(int dim, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 构建 upsert 用的 JSON 文档
static rapidjson::Document MakeUpsertDoc(const std::vector<float>& vec, uint64_t id) {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
    rapidjson::Value vec_array(rapidjson::kArrayType);
    for (float v : vec) {
        vec_array.PushBack(v, alloc);
    }
    doc.AddMember("vectors", vec_array, alloc);
    doc.AddMember("id", id, alloc);
    return doc;
}

// 构建 search 用的 JSON 请求
static rapidjson::Document MakeSearchDoc(const std::vector<float>& query, int k, const char* index_type = "FLAT") {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
    rapidjson::Value vec_array(rapidjson::kArrayType);
    for (float v : query) {
        vec_array.PushBack(v, alloc);
    }
    doc.AddMember("vectors", vec_array, alloc);
    doc.AddMember("k", k, alloc);
    doc.AddMember("indexType", rapidjson::Value(index_type, alloc), alloc);
    return doc;
}

// 测试 1：创建不同维度的 Collection 并验证隔离性
TEST(MultiCollectionTest, DifferentDimensionIsolation) {
    auto& cm = CollectionManager::Instance();

    // 创建一个 64 维的 Collection
    ASSERT_TRUE(cm.CreateCollection("dim64", 64, 10000));
    ASSERT_TRUE(cm.HasCollection("dim64"));

    // 创建一个 256 维的 Collection
    ASSERT_TRUE(cm.CreateCollection("dim256", 256, 10000));
    ASSERT_TRUE(cm.HasCollection("dim256"));

    // default Collection 应该是 128 维（来自 config）
    ASSERT_TRUE(cm.HasCollection("default"));

    VectorDatabase db("/root/test_vectordb/storage", "/root/test_vectordb/wal/wal.log");

    // 在 dim64 中插入 64 维向量
    auto v64 = GenVec(64, 42);
    auto doc64 = MakeUpsertDoc(v64, 100);
    db.Upsert("dim64", 100, doc64, IndexFactory::IndexType::FLAT);

    // 在 dim256 中插入 256 维向量
    auto v256 = GenVec(256, 99);
    auto doc256 = MakeUpsertDoc(v256, 200);
    db.Upsert("dim256", 200, doc256, IndexFactory::IndexType::FLAT);

    // 在 default 中插入 128 维向量
    auto v128 = GenVec(128, 7);
    auto doc128 = MakeUpsertDoc(v128, 300);
    db.Upsert("default", 300, doc128, IndexFactory::IndexType::FLAT);

    // 搜索 dim64，应该找到 id=100
    auto req64 = MakeSearchDoc(v64, 1);
    auto [ids64, dists64] = db.Search("dim64", req64);
    ASSERT_EQ(ids64.size(), 1u);
    EXPECT_EQ(ids64[0], 100);

    // 搜索 dim256，应该找到 id=200
    auto req256 = MakeSearchDoc(v256, 1);
    auto [ids256, dists256] = db.Search("dim256", req256);
    ASSERT_EQ(ids256.size(), 1u);
    EXPECT_EQ(ids256[0], 200);

    // 搜索 default，应该找到 id=300
    auto req128 = MakeSearchDoc(v128, 1);
    auto [ids128, dists128] = db.Search("default", req128);
    ASSERT_EQ(ids128.size(), 1u);
    EXPECT_EQ(ids128[0], 300);

    // 清理
    cm.DropCollection("dim64");
    cm.DropCollection("dim256");
}

// 测试 2：default Collection 向后兼容
TEST(MultiCollectionTest, DefaultCollectionBackwardCompat) {
    auto& cm = CollectionManager::Instance();
    ASSERT_TRUE(cm.HasCollection("default"));

    CollectionMeta meta;
    ASSERT_TRUE(cm.GetCollectionMeta("default", meta));
    EXPECT_EQ(meta.dimension, 128);  // 来自 config DIM=128
    EXPECT_EQ(meta.name, "default");

    VectorDatabase db("/root/test_vectordb/storage", "/root/test_vectordb/wal/wal.log");

    // 在 default Collection 中插入和搜索
    auto vec = GenVec(128, 123);
    auto doc = MakeUpsertDoc(vec, 999);
    db.Upsert("default", 999, doc, IndexFactory::IndexType::FLAT);

    auto req = MakeSearchDoc(vec, 1);
    auto [ids, dists] = db.Search("default", req);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 999);
}

// 测试 3：DropCollection
TEST(MultiCollectionTest, DropCollection) {
    auto& cm = CollectionManager::Instance();

    ASSERT_TRUE(cm.CreateCollection("temp_coll", 32, 1000));
    ASSERT_TRUE(cm.HasCollection("temp_coll"));

    // 不能删除 default
    ASSERT_FALSE(cm.DropCollection("default"));

    // 删除 temp_coll
    ASSERT_TRUE(cm.DropCollection("temp_coll"));
    ASSERT_FALSE(cm.HasCollection("temp_coll"));

    // 删除不存在的 Collection
    ASSERT_FALSE(cm.DropCollection("nonexistent"));
}

// 测试 4：重复创建 Collection 应失败
TEST(MultiCollectionTest, DuplicateCreateFails) {
    auto& cm = CollectionManager::Instance();
    ASSERT_TRUE(cm.CreateCollection("dup_test", 64, 1000));
    // 再次创建同名应失败
    ASSERT_FALSE(cm.CreateCollection("dup_test", 64, 1000));
    ASSERT_FALSE(cm.CreateCollection("dup_test", 128, 1000));
    cm.DropCollection("dup_test");
}

// 测试 5：ListCollections 包含 default
TEST(MultiCollectionTest, ListCollections) {
    auto& cm = CollectionManager::Instance();
    auto names = cm.ListCollections();
    // default 应该在列表中
    bool has_default = false;
    for (const auto& n : names) {
        if (n == "default") has_default = true;
    }
    EXPECT_TRUE(has_default);
}

// 测试 6：Collection 元数据 Save/Load
TEST(MultiCollectionTest, MetaSaveLoad) {
    auto& cm = CollectionManager::Instance();

    cm.CreateCollection("save_test_64", 64, 5000);
    cm.CreateCollection("save_test_256", 256, 5000);

    // 保存到临时文件
    std::string meta_path = "/tmp/test_collections_meta.json";
    cm.SaveToDisk(meta_path);

    // 验证文件存在
    std::ifstream ifs(meta_path);
    ASSERT_TRUE(ifs.is_open());
    ifs.close();

    // 删除 Collection 后重新加载
    cm.DropCollection("save_test_64");
    cm.DropCollection("save_test_256");
    ASSERT_FALSE(cm.HasCollection("save_test_64"));

    cm.LoadFromDisk(meta_path);
    ASSERT_TRUE(cm.HasCollection("save_test_64"));
    ASSERT_TRUE(cm.HasCollection("save_test_256"));

    CollectionMeta meta64;
    ASSERT_TRUE(cm.GetCollectionMeta("save_test_64", meta64));
    EXPECT_EQ(meta64.dimension, 64);

    CollectionMeta meta256;
    ASSERT_TRUE(cm.GetCollectionMeta("save_test_256", meta256));
    EXPECT_EQ(meta256.dimension, 256);

    // 清理
    cm.DropCollection("save_test_64");
    cm.DropCollection("save_test_256");
    std::remove(meta_path.c_str());
}

// 测试 7：在非 default Collection 上使用不同索引类型
TEST(MultiCollectionTest, DifferentIndexTypePerCollection) {
    auto& cm = CollectionManager::Instance();
    ASSERT_TRUE(cm.CreateCollection("sq8_coll", 128, 10000));

    VectorDatabase db("/root/test_vectordb/storage", "/root/test_vectordb/wal/wal.log");

    // 批量插入到 sq8_coll 使用 SQ8 索引
    std::vector<uint64_t> ids;
    std::vector<rapidjson::Document> docs;
    for (int i = 0; i < 50; ++i) {
        ids.push_back(1000 + i);
        docs.push_back(MakeUpsertDoc(GenVec(128, i), 1000 + i));
    }
    db.BatchUpsert("sq8_coll", ids, docs, IndexFactory::IndexType::SQ8);

    // 搜索 SQ8 索引
    auto query = GenVec(128, 0);
    auto req = MakeSearchDoc(query, 5, "SQ8");
    req.AddMember("collectionName", "sq8_coll", req.GetAllocator());
    auto [result_ids, result_dists] = db.Search("sq8_coll", req);
    // 应该能找到结果（至少 id=1000，因为 query 和 id=1000 用同一个 seed）
    ASSERT_GE(result_ids.size(), 1u);

    cm.DropCollection("sq8_coll");
}

// 测试 8：自动创建 Collection（不预先 Create，直接写入）
TEST(MultiCollectionTest, AutoCreateOnWrite) {
    auto& cm = CollectionManager::Instance();
    // 确保不存在
    if (cm.HasCollection("auto_256")) cm.DropCollection("auto_256");
    ASSERT_FALSE(cm.HasCollection("auto_256"));

    VectorDatabase db("/root/test_vectordb/storage", "/root/test_vectordb/wal/wal.log");

    // 单条 Upsert 自动创建 256 维 Collection
    auto v256 = GenVec(256, 42);
    auto doc = MakeUpsertDoc(v256, 1001);
    db.Upsert("auto_256", 1001, doc, IndexFactory::IndexType::FLAT);

    ASSERT_TRUE(cm.HasCollection("auto_256"));
    CollectionMeta meta;
    ASSERT_TRUE(cm.GetCollectionMeta("auto_256", meta));
    EXPECT_EQ(meta.dimension, 256);

    // 搜索应能找到插入的向量
    auto req = MakeSearchDoc(v256, 1);
    auto [ids, dists] = db.Search("auto_256", req);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 1001);

    cm.DropCollection("auto_256");
}

// 测试 9：BatchUpsert 自动创建 Collection
TEST(MultiCollectionTest, AutoCreateOnBatchWrite) {
    auto& cm = CollectionManager::Instance();
    std::string coll_name = "batch_auto_64";
    if (cm.HasCollection(coll_name)) cm.DropCollection(coll_name);
    ASSERT_FALSE(cm.HasCollection(coll_name));

    VectorDatabase db("/root/test_vectordb/storage", "/root/test_vectordb/wal/wal.log");

    // 批量写入 64 维向量（不预先创建 Collection）
    std::vector<uint64_t> ids;
    std::vector<rapidjson::Document> docs;
    for (int i = 0; i < 10; ++i) {
        ids.push_back(2000 + i);
        docs.push_back(MakeUpsertDoc(GenVec(64, i), 2000 + i));
    }
    db.BatchUpsert(coll_name, ids, docs, IndexFactory::IndexType::FLAT);

    ASSERT_TRUE(cm.HasCollection(coll_name));
    CollectionMeta meta;
    ASSERT_TRUE(cm.GetCollectionMeta(coll_name, meta));
    EXPECT_EQ(meta.dimension, 64);

    // 搜索应能找到
    auto query = GenVec(64, 0);
    auto req = MakeSearchDoc(query, 3, "FLAT");
    auto [result_ids, result_dists] = db.Search(coll_name, req);
    ASSERT_GE(result_ids.size(), 1u);
    EXPECT_EQ(result_ids[0], 2000);

    cm.DropCollection(coll_name);
}

}  // namespace vectordb
