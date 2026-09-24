// IndexFactory used_types_ 回归测试
//
// 背景：每个 collection 会预建 10 种索引，但业务写入通常只命中其中一种。
// 快照若全量落盘，单次要写 >1GB，而快照在 NuRaft commit 线程上同步执行，
// 会阻塞提交十余秒（超过 client_req_timeout_ 即写入失败）。
//
// 覆盖四点：
//   1. 只标记的类型才落盘
//   2. used_types.bin sidecar 持久化往返
//   3. 旧格式快照（无 sidecar）兜底为全类型，加载后剪枝无数据的类型
//   4. GetTotalCount 按 used_types_ 选路——旧实现硬编码 FLAT，
//      业务用 HNSW/GARDEN 时会命中 FLAT 的空索引恒返回 0，
//      导致负载上报（vectorCount）与再平衡决策失效
//
// 运行：./test/index_factory_used_types_test

#include "index/index_factory.h"
#include "index/hnswlib_index.h"
#include "common/vector_init.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

namespace vectordb {

class UsedTypesTestEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const used_types_env =
    ::testing::AddGlobalTestEnvironment(new UsedTypesTestEnv);

namespace {

constexpr int kDim = 8;
constexpr int kHnswIdx = static_cast<int>(IndexFactory::IndexType::HNSW);   // 1
constexpr int kFilterIdx = static_cast<int>(IndexFactory::IndexType::FILTER); // 2
constexpr int kFlatIdx = static_cast<int>(IndexFactory::IndexType::FLAT);   // 0

auto TempPrefix(const std::string& tag) -> std::string {
    const auto dir = std::filesystem::temp_directory_path() / ("used_types_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return (dir / "coll_").string();
}

auto DirOf(const std::string& prefix) -> std::filesystem::path {
    return std::filesystem::path(prefix).parent_path();
}

auto CountFiles(const std::filesystem::path& dir) -> int {
    int n = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file()) ++n;
    }
    return n;
}

auto GenVec(int seed) -> std::vector<float> {
    std::vector<float> v(kDim);
    for (int i = 0; i < kDim; ++i) v[i] = 0.1f * (seed + i);
    return v;
}

void InsertIntoHnsw(IndexFactory& f, int n) {
    auto* hnsw = static_cast<HNSWLibIndex*>(f.GetIndex(IndexFactory::IndexType::HNSW));
    for (int i = 0; i < n; ++i) {
        hnsw->InsertVectors(GenVec(i), i);
    }
}

}  // namespace

// ============================================================================
// 用例 1：只标记的类型才落盘
// ============================================================================

TEST(UsedTypes, SaveOnlyMarkedTypes) {
    const std::string prefix = TempPrefix("save_only_marked");

    IndexFactory f;
    f.Init(IndexFactory::IndexType::HNSW, kDim, 1000);
    f.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    f.Init(IndexFactory::IndexType::FLAT, kDim, 1000);

    // 只写 HNSW。未写入的 HNSW 之外的类型不应落盘。
    InsertIntoHnsw(f, 5);
    f.MarkUsed(IndexFactory::IndexType::HNSW);

    f.SaveIndex(prefix);

    const auto dir = DirOf(prefix);
    // HNSW .index + FILTER .index（无条件保留）+ used_types.bin = 3 个文件
    EXPECT_EQ(CountFiles(dir), 3) << "未使用的索引类型不应落盘";
    EXPECT_TRUE(std::filesystem::exists(prefix + std::to_string(kHnswIdx) + ".index"));
    EXPECT_TRUE(std::filesystem::exists(prefix + std::to_string(kFilterIdx) + ".index"));
    EXPECT_TRUE(std::filesystem::exists(prefix + "used_types.bin"));
    EXPECT_FALSE(std::filesystem::exists(prefix + std::to_string(kFlatIdx) + ".index"))
        << "从未写入的 FLAT 不应落盘";

    std::filesystem::remove_all(dir);
}

// ============================================================================
// 用例 2：used_types.bin sidecar 持久化往返
// ============================================================================

TEST(UsedTypes, SidecarRoundTrip) {
    const std::string prefix = TempPrefix("roundtrip");

    IndexFactory f;
    f.Init(IndexFactory::IndexType::GARDEN_HNSW, kDim, 1000);
    f.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    f.MarkUsed(IndexFactory::IndexType::GARDEN_HNSW);
    f.SaveIndex(prefix);

    IndexFactory g;
    g.Init(IndexFactory::IndexType::GARDEN_HNSW, kDim, 1000);
    g.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    g.LoadIndex(prefix);

    const auto used = g.GetUsedTypes();
    EXPECT_EQ(used.count(IndexFactory::IndexType::GARDEN_HNSW), 1);
    EXPECT_EQ(used.count(IndexFactory::IndexType::FILTER), 1);
    EXPECT_EQ(used.count(IndexFactory::IndexType::FLAT), 0) << "未标记的类型不应被恢复";

    std::filesystem::remove_all(DirOf(prefix));
}

// ============================================================================
// 用例 3：旧格式快照（无 sidecar）兜底为全类型，加载后剪枝无数据的类型
// ============================================================================

TEST(UsedTypes, LegacySnapshotFallbackAndPrune) {
    const std::string prefix = TempPrefix("legacy");

    // 造一个"有 HNSW 数据"的旧快照，然后删掉 sidecar 模拟旧格式
    IndexFactory f;
    f.Init(IndexFactory::IndexType::HNSW, kDim, 1000);
    f.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    f.Init(IndexFactory::IndexType::FLAT, kDim, 1000);
    f.MarkUsed(IndexFactory::IndexType::HNSW);
    InsertIntoHnsw(f, 20);
    f.SaveIndex(prefix);
    std::filesystem::remove(prefix + "used_types.bin");

    IndexFactory g;
    g.Init(IndexFactory::IndexType::HNSW, kDim, 1000);
    g.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    g.Init(IndexFactory::IndexType::FLAT, kDim, 1000);
    g.LoadIndex(prefix);  // 无 sidecar → 先全量加载，再剪枝空类型

    const auto used = g.GetUsedTypes();
    EXPECT_EQ(used.count(IndexFactory::IndexType::HNSW), 1) << "有数据的类型必须保留";
    EXPECT_EQ(used.count(IndexFactory::IndexType::FILTER), 1) << "FILTER 永远保留";
    EXPECT_EQ(used.count(IndexFactory::IndexType::FLAT), 0) << "无数据的类型应被剪枝";

    // 剪枝结果应立即固化，避免下次快照退回全量落盘
    EXPECT_TRUE(std::filesystem::exists(prefix + "used_types.bin"));

    // 数据完整性：HNSW 的 20 条应能正确加载
    auto* hnsw = static_cast<HNSWLibIndex*>(g.GetIndex(IndexFactory::IndexType::HNSW));
    ASSERT_NE(hnsw, nullptr);
    EXPECT_EQ(hnsw->GetTotalCount(), 20);

    std::filesystem::remove_all(DirOf(prefix));
}

// ============================================================================
// 用例 4：GetTotalCount 按 used_types_ 选路
// ============================================================================

TEST(UsedTypes, GetTotalCountRoutesByUsedTypes) {
    const std::string prefix = TempPrefix("count");

    IndexFactory f;
    f.Init(IndexFactory::IndexType::HNSW, kDim, 1000);
    f.Init(IndexFactory::IndexType::FILTER, kDim, 1000);
    f.Init(IndexFactory::IndexType::FLAT, kDim, 1000);

    // 只写 HNSW。旧实现硬编码读 FLAT，会命中空索引恒返回 0，
    // 导致负载上报（vectorCount）与再平衡决策失效。
    InsertIntoHnsw(f, 7);
    f.MarkUsed(IndexFactory::IndexType::HNSW);

    EXPECT_EQ(f.GetTotalCount(), 7) << "应从实际被写入的 HNSW 取计数，而非空的 FLAT";

    std::filesystem::remove_all(DirOf(prefix));
}

}  // namespace vectordb
