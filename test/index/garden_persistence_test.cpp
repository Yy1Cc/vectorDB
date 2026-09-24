// GARDEN 持久化与过滤正确性回归测试
//
// 覆盖三个此前"静默失败"的缺陷（失败时都不报错，只是结果不对或数据丢失）：
//
//   1. IndexFactory::SaveIndex 缺 GARDEN_HNSW 分支
//      → 快照时 GARDEN 索引一个字节都不落盘，重启后 GardenIndex::LoadIndex
//        找不到快照文件直接跳过，表现为数据全丢、查询返回全 -1。
//
//   2. 多文件格式（每个离散值 / 每个分桶各一个 .index 文件）
//      → 高基数字段会产生海量小文件。已改为单文件容器格式，
//        本测试校验落盘后只有一个文件。
//
//   3. GardenFilter::has_range 只写不读
//      → 对已注册的连续字段做等值过滤时 range_min/max 保持默认 0,0，
//        桶收集条件退化成"只选第 0 号桶"，召回严重劣化。
//
// 运行：./test/garden_persistence_test

#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "roaring/roaring.h"
#include "common/vector_init.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace vectordb {

class GardenPersistenceTestEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment *const garden_persist_env =
    ::testing::AddGlobalTestEnvironment(new GardenPersistenceTestEnv);

namespace {

constexpr int kDim = 8;

auto GenVec(int seed) -> std::vector<float> {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(kDim);
    for (auto &x : v) x = dist(rng);
    return v;
}

auto MakeBitmap(const std::vector<uint32_t> &ids) -> roaring_bitmap_t * {
    roaring_bitmap_t *bm = roaring_bitmap_create();
    for (auto id : ids) roaring_bitmap_add(bm, id);
    return bm;
}

// 快照落盘目录：每个用例用独立子目录，避免互相干扰
auto SnapshotPrefix(const std::string &tag) -> std::string {
    const auto dir = std::filesystem::temp_directory_path() / ("garden_persist_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return (dir / "coll_").string();
}

// 构造一个同时具备离散字段与连续字段的索引。
// price 共 10 个分桶（[0,999] ... [9000,9999]），用于验证桶路由。
void Populate(GardenIndex *garden, int n) {
    garden->RegisterDiscreteField("brand");
    // 5000 条数据下，相对暴力阈值 = max(1% * 5000, 500) = 500，
    // 因此候选集 > 500 时会走子图路径而非暴力路径。
    garden->RegisterContinuousField("price", 0, 10000, 1000);

    for (int i = 0; i < n; ++i) {
        const int64_t price = i * 2;  // 0 .. 9998，均匀铺满 10 个桶
        std::string brand = (i % 3 == 0) ? "apple" : (i % 3 == 1) ? "xiaomi" : "huawei";
        garden->Insert(GenVec(i), i, {{"price", price}}, {{"brand", brand}});
    }
}

}  // namespace

// ============================================================================
// 用例 1：SaveIndex → LoadIndex 往返一致性
// ============================================================================

TEST(GardenPersistence, SaveLoadRoundTrip) {
    const std::string prefix = SnapshotPrefix("roundtrip");
    constexpr int kN = 2000;

    GardenIndex origin(kDim, kN * 2);
    Populate(&origin, kN);
    ASSERT_EQ(origin.GetTotalCount(), static_cast<size_t>(kN));

    origin.SaveIndex(prefix);

    GardenIndex restored(kDim, kN * 2);
    restored.LoadIndex(prefix);

    EXPECT_EQ(restored.GetTotalCount(), origin.GetTotalCount());
    EXPECT_TRUE(restored.HasDiscreteSubgraph("brand"));
    EXPECT_TRUE(restored.HasContinuousSubgraph("price"));
    EXPECT_EQ(restored.GetDiscreteValues("brand").size(), 3);

    // 恢复后搜索结果应与原索引一致
    const int kTarget = 777;
    auto query = GenVec(kTarget);
    auto *bm = MakeBitmap({static_cast<uint32_t>(kTarget)});

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {(kTarget % 3 == 0) ? "apple" : (kTarget % 3 == 1) ? "xiaomi" : "huawei"};
    gf.has_range = false;
    gf.bitmap = bm;
    gf.cardinality = 1;

    auto [origin_ids, origin_dists] = origin.Search(query, 5, {gf}, nullptr, 50);
    auto [restored_ids, restored_dists] = restored.Search(query, 5, {gf}, nullptr, 50);

    ASSERT_FALSE(origin_ids.empty());
    EXPECT_EQ(origin_ids[0], kTarget);
    EXPECT_EQ(restored_ids[0], origin_ids[0]);
    EXPECT_FLOAT_EQ(restored_dists[0], origin_dists[0]);

    roaring_bitmap_free(bm);
    std::filesystem::remove_all(std::filesystem::path(prefix).parent_path());
}

// ============================================================================
// 用例 2：快照是单个文件（不是一值一文件）
// ============================================================================

TEST(GardenPersistence, SnapshotIsSingleFile) {
    const std::string prefix = SnapshotPrefix("singlefile");
    constexpr int kN = 600;

    GardenIndex garden(kDim, kN * 2);
    Populate(&garden, kN);
    garden.SaveIndex(prefix);

    const auto dir = std::filesystem::path(prefix).parent_path();
    int file_count = 0;
    int garden_bin_count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        ++file_count;
        if (entry.path().filename().string() == "coll__garden.bin") ++garden_bin_count;
    }

    // 3 个离散值 + 10 个分桶，旧格式会产生 14 个文件；新格式只有 1 个。
    EXPECT_EQ(file_count, 1) << "快照目录应只包含一个容器文件，实际有 " << file_count << " 个";
    EXPECT_EQ(garden_bin_count, 1);

    std::filesystem::remove_all(dir);
}

// ============================================================================
// 用例 3：连续字段等值过滤不再退化成只搜第 0 号桶
// ============================================================================

// has_range 为 true：正常的区间过滤，应命中 [4000,4999] 与 [5000,5999] 两个桶
TEST(GardenFilterRange, ContinuousRangeFilterHitsCorrectBuckets) {
    constexpr int kN = 5000;
    GardenIndex garden(kDim, kN * 2);
    Populate(&garden, kN);

    // price = i * 2，落在 [4000, 5999] 的是 i ∈ [2000, 2999]
    std::vector<uint32_t> ids;
    for (int i = 2000; i < 3000; ++i) ids.push_back(static_cast<uint32_t>(i));
    auto *bm = MakeBitmap(ids);

    GardenFilter gf;
    gf.field = "price";
    gf.has_range = true;
    gf.range_min = 4000;
    gf.range_max = 5999;
    gf.bitmap = bm;
    gf.cardinality = roaring_bitmap_get_cardinality(bm);

    const int kTarget = 2500;  // price = 5000，落在查询区间内
    auto [out_ids, out_dists] = garden.Search(GenVec(kTarget), 10, {gf}, nullptr, 50);

    ASSERT_FALSE(out_ids.empty());
    EXPECT_NE(out_ids[0], -1) << "区间过滤未返回任何结果";
    EXPECT_EQ(out_ids[0], kTarget);

    roaring_bitmap_free(bm);
}

// has_range 为 false（等值过滤的历史写法）：修复前 range 保持默认 (0,0)，
// 只收集到第 0 号桶，导致区间内的向量一个都搜不到；修复后该条件不再当 Pruner，
// 退化为全量图 + bitmap 过滤，结果正确。
TEST(GardenFilterRange, ContinuousFilterWithoutRangeStillReturnsResults) {
    constexpr int kN = 5000;
    GardenIndex garden(kDim, kN * 2);
    Populate(&garden, kN);

    std::vector<uint32_t> ids;
    for (int i = 2000; i < 3000; ++i) ids.push_back(static_cast<uint32_t>(i));
    auto *bm = MakeBitmap(ids);

    GardenFilter gf;
    gf.field = "price";
    gf.has_range = false;  // 关键：不提供区间信息
    gf.bitmap = bm;
    gf.cardinality = roaring_bitmap_get_cardinality(bm);

    const int kTarget = 2500;
    auto [out_ids, out_dists] = garden.Search(GenVec(kTarget), 10, {gf}, nullptr, 50);

    ASSERT_FALSE(out_ids.empty());
    EXPECT_NE(out_ids[0], -1) << "无区间信息时不应退化成只搜第 0 号桶";
    EXPECT_EQ(out_ids[0], kTarget);

    roaring_bitmap_free(bm);
}

// ============================================================================
// 用例 4：删除后落盘，重启恢复不应"复活"被删数据
// ============================================================================

TEST(GardenPersistence, DeletedVectorsStayDeletedAfterReload) {
    const std::string prefix = SnapshotPrefix("deleted");
    constexpr int kN = 800;

    GardenIndex garden(kDim, kN * 2);
    Populate(&garden, kN);
    garden.RemoveVectors({1, 2, 3});
    EXPECT_EQ(garden.GetTotalCount(), static_cast<size_t>(kN - 3));

    garden.SaveIndex(prefix);

    GardenIndex restored(kDim, kN * 2);
    restored.LoadIndex(prefix);
    EXPECT_EQ(restored.GetTotalCount(), static_cast<size_t>(kN - 3));

    std::filesystem::remove_all(std::filesystem::path(prefix).parent_path());
}

}  // namespace vectordb
