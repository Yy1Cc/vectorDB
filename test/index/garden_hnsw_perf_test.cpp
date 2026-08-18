// perf 热点定位目标：分别压 HNSW post-filter（低选择性断崖）和 GARDEN 子图搜索
// 用 perf record 分别采样两个 gtest，对比两条路径的调用栈热点
//
// 运行：
//   perf record -g ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.HnswPostFilter
//   perf record -g ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.GardenSubgraph

#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "roaring/roaring.hh"
#include <vector>
#include <string>
#include <map>
#include <cstdio>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class PerfEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const g_perf_env = ::testing::AddGlobalTestEnvironment(new PerfEnv);

static const int DIM = 128;
static const int N = 100000;
static const int REPEAT = 5000;  // 搜索循环次数，让搜索时间占主导便于 perf 采样

static std::vector<float> GenVec(int i) {
    std::vector<float> v(DIM);
    for (int d = 0; d < DIM; ++d) v[d] = (i + d) * 0.001f;
    return v;
}

// HNSW post-filter：2% 低选择性（K=50），对应断崖场景
TEST(PerfHotspot, HnswPostFilter) {
    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));
    for (int i = 0; i < N; ++i) data[i] = GenVec(i);
    std::vector<float> query = GenVec(12345);

    HNSWLibIndex hnsw(DIM, N, IndexFactory::MetricType::L2, 16, 200);
    for (int i = 0; i < N; ++i) hnsw.InsertVectors(data[i], i);
    std::fprintf(stderr, "[perf] HNSW built\n");

    // brand=brand_5 的 bitmap（K=50，candidate=2000，2% 选择性）
    const int K = 50, target_mod = 5;
    roaring_bitmap_t* bm = roaring_bitmap_create();
    for (int i = 0; i < N; ++i)
        if (i % K == target_mod) roaring_bitmap_add(bm, static_cast<uint32_t>(i));

    std::fprintf(stderr, "[perf] HNSW post-filter search start (repeat=%d)\n", REPEAT);
    int64_t acc = 0;
    for (int r = 0; r < REPEAT; ++r) {
        auto [ids, dists] = hnsw.SearchVectors(query, 10, bm, 50);
        acc += ids.empty() ? 0 : ids[0];
    }
    std::fprintf(stderr, "[perf] HNSW done acc=%lld\n", (long long)acc);
    roaring_bitmap_free(bm);
}

// GARDEN 子图搜索：10% 选择性（K=10，candidate=10000 走子图），对应优势场景
TEST(PerfHotspot, GardenSubgraph) {
    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));
    for (int i = 0; i < N; ++i) data[i] = GenVec(i);
    std::vector<float> query = GenVec(12345);

    const int K = 10, target_mod = 5;
    std::string target = "brand_5";

    GardenIndex garden(DIM, N, IndexFactory::MetricType::L2);
    garden.RegisterDiscreteField("brand");
    for (int i = 0; i < N; ++i) {
        std::map<std::string, int64_t> int_fields;
        std::map<std::string, std::string> str_fields = {{"brand", "brand_" + std::to_string(i % K)}};
        garden.Insert(data[i], i, int_fields, str_fields);
    }
    std::fprintf(stderr, "[perf] GARDEN built\n");

    roaring_bitmap_t* bm = roaring_bitmap_create();
    for (int i = 0; i < N; ++i)
        if (i % K == target_mod) roaring_bitmap_add(bm, static_cast<uint32_t>(i));

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {target};
    gf.bitmap = bm;
    gf.cardinality = roaring_bitmap_get_cardinality(bm);

    std::fprintf(stderr, "[perf] GARDEN subgraph search start (repeat=%d)\n", REPEAT);
    int64_t acc = 0;
    for (int r = 0; r < REPEAT; ++r) {
        auto [ids, dists] = garden.Search(query, 10, {gf}, nullptr, 50);
        acc += ids.empty() ? 0 : ids[0];
    }
    std::fprintf(stderr, "[perf] GARDEN done acc=%lld\n", (long long)acc);
    roaring_bitmap_free(bm);
}

}  // namespace vectordb
