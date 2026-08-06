// GARDEN vs HNSW 过滤搜索性能对比压测：验证 GARDEN 解决 HNSW 低选择性断崖
//
// 对比路径：
//   HNSW：全量图 + bitmap 后过滤（RoaringBitmapIDFilter 白名单，低选择性遍历大量被过滤节点 → 断崖）
//   GARDEN：子图路由（低选择性走暴力/子图搜索，避开全量图无效遍历）
//
// 选择性梯度：K 个 brand 值均匀分布，查询 brand=brand_5，选择性 = 1/K
//   K=2 → 50%（candidate=50000），K=10 → 10%（candidate=10000），K=50 → 2%（candidate=2000）
// GARDEN 子图路径触发条件：candidate >= kBruteBound(10000) 且 selectivity < kFullSearchRate(0.8)
//
// 运行：./test/garden_vs_hnsw_bench_test

#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "roaring/roaring.hh"
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class GardenVsHnswEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const gvs_hnsw_env = ::testing::AddGlobalTestEnvironment(new GardenVsHnswEnv);

static const int DIM = 128;
static const int N = 100000;

static std::vector<float> GenVec(int i) {
    std::vector<float> v(DIM);
    for (int d = 0; d < DIM; ++d) v[d] = (i + d) * 0.001f;
    return v;
}

using Clock = std::chrono::high_resolution_clock;

// 暴力 ground truth：在 brand==target（i%K==target_mod）的 id 里算距离取 top-k
static std::vector<int64_t> BruteGT(const std::vector<std::vector<float>>& data,
                                    const std::vector<float>& query, int k,
                                    int target_mod, int K) {
    std::vector<std::pair<float, int64_t>> cand;
    cand.reserve(N / K + 1);
    for (int i = 0; i < N; ++i) {
        if (i % K != target_mod) continue;
        float dist = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            float diff = query[d] - data[i][d];
            dist += diff * diff;
        }
        cand.emplace_back(dist, i);
    }
    std::partial_sort(cand.begin(), cand.begin() + std::min<int>(k, cand.size()), cand.end());
    std::vector<int64_t> gt;
    for (int i = 0; i < std::min<int>(k, cand.size()); ++i) gt.push_back(cand[i].second);
    return gt;
}

static double Recall(const std::vector<int64_t>& got, const std::vector<int64_t>& gt) {
    int hit = 0;
    for (int64_t id : got)
        for (int64_t g : gt)
            if (id == g) ++hit;
    return gt.empty() ? 0.0 : static_cast<double>(hit) / gt.size();
}

// 测量 QPS + p50/p99 + 召回
struct BenchResult {
    double qps;
    double p50_ms;
    double p99_ms;
    double recall;
};

template <typename SearchFn>
static BenchResult RunBench(SearchFn&& search, const std::vector<float>& query, int k,
                            const std::vector<int64_t>& gt, int repeat = 200) {
    std::vector<double> times;
    times.reserve(repeat);
    double recall_sum = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto t0 = Clock::now();
        auto [ids, dists] = search();
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        recall_sum += Recall(ids, gt);
    }
    std::sort(times.begin(), times.end());
    double total = std::accumulate(times.begin(), times.end(), 0.0);
    BenchResult r;
    r.qps = repeat / (total / 1000.0);
    r.p50_ms = times[times.size() / 2];
    r.p99_ms = times[static_cast<size_t>(times.size() * 0.99)];
    r.recall = recall_sum / repeat;
    return r;
}

TEST(GardenVsHnswBench, SelectivityGradient) {
    // 生成数据（确定性 pattern）
    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));
    for (int i = 0; i < N; ++i) data[i] = GenVec(i);

    std::vector<float> query = GenVec(12345);

    // HNSW 全量图只构建一次（不含 brand 信息，bitmap 仅查询时用）
    auto th0 = Clock::now();
    HNSWLibIndex hnsw(DIM, N, IndexFactory::MetricType::L2, 16, 200);
    for (int i = 0; i < N; ++i) hnsw.InsertVectors(data[i], i);
    auto th1 = Clock::now();
    std::printf("[build] HNSW full graph: %.1fs\n",
                std::chrono::duration<double>(th1 - th0).count());

    std::printf("\n==== GARDEN vs HNSW 过滤搜索对比 (N=%d, dim=%d, k=10) ====\n", N, DIM);
    std::printf("| 选择性 | candidate | HNSW QPS | GARDEN QPS | HNSW p50 | GARDEN p50 | HNSW召回 | GARDEN召回 | GARDEN/HNSW |\n");
    std::printf("|--------|-----------|----------|------------|----------|------------|----------|------------|-------------|\n");

    for (int K : {2, 10, 50}) {
        int target_mod = 5 % K;
        std::string target = "brand_" + std::to_string(target_mod);

        // 构建 brand=target 的 bitmap（白名单）
        roaring_bitmap_t* bm = roaring_bitmap_create();
        for (int i = 0; i < N; ++i)
            if (i % K == target_mod) roaring_bitmap_add(bm, static_cast<uint32_t>(i));
        uint64_t candidate = roaring_bitmap_get_cardinality(bm);
        double selectivity = 1.0 / K;

        // GARDEN 构建（按 brand 建子图）
        auto tg0 = Clock::now();
        GardenIndex garden(DIM, N, IndexFactory::MetricType::L2);
        garden.RegisterDiscreteField("brand");
        for (int i = 0; i < N; ++i) {
            std::map<std::string, int64_t> int_fields;
            std::map<std::string, std::string> str_fields = {{"brand", "brand_" + std::to_string(i % K)}};
            garden.Insert(data[i], i, int_fields, str_fields);
        }
        auto tg1 = Clock::now();
        double garden_build = std::chrono::duration<double>(tg1 - tg0).count();

        // ground truth
        std::vector<int64_t> gt = BruteGT(data, query, 10, target_mod, K);

        // HNSW：全量图 + bitmap 后过滤
        BenchResult rh = RunBench(
            [&]() { return hnsw.SearchVectors(query, 10, bm, 50); }, query, 10, gt);

        // GARDEN：子图路由
        GardenFilter gf;
        gf.field = "brand";
        gf.discrete_values = {target};
        gf.bitmap = bm;
        gf.cardinality = candidate;
        BenchResult rg = RunBench(
            [&]() { return garden.Search(query, 10, {gf}, nullptr, 50); }, query, 10, gt);

        std::printf("| %4.0f%%  | %9llu | %8.0f | %10.0f | %7.2fms | %9.2fms | %8.3f | %10.3f | %10.1fx |\n",
                    selectivity * 100, (unsigned long long)candidate,
                    rh.qps, rg.qps, rh.p50_ms, rg.p50_ms, rh.recall, rg.recall, rg.qps / rh.qps);
        std::printf("         [garden build: %.1fs]\n", garden_build);

        roaring_bitmap_free(bm);
    }
    std::printf("=========================================================================\n");
    std::printf("GARDEN/HNSW > 1 表示 GARDEN 更快。低选择性(2%%)时 HNSW post-filter 应断崖。\n");
}

}  // namespace vectordb
