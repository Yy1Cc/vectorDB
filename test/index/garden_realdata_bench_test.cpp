// GARDEN vs 暴力 vs HNSW 中间过滤 —— 真实数据集对比（SPCL/arxiv-for-fanns-medium）
//
// 与 garden_vs_hnsw_bench_test.cpp（人造线性数据，recall 失真为 1.0）的区别：
//   本测试使用 100,000 条 4096 维真实 arXiv 论文 embedding（stella_en_400M_v5），
//   索引近似性在真实数据分布下暴露，recall 是诚实指标。
//
// 三路方法：
//   1. GARDEN        ：子图路由（离散按值子图 / 连续分桶子图，按选择性路由）
//   2. 暴力           ：遍历过滤位图候选、逐个算 L2 取 top-k（精确下界，recall=1.0 基准）
//   3. HNSW 中间过滤  ：searchKnn + RoaringBitmapIDFilter，图遍历中判白名单（项目现有实现）
//
// 两类过滤：
//   EM：query label -> 字段 number_of_sub_categories（离散子图路径）
//   R ：query [range_start, range_end] -> 字段 update_date（连续分桶子图路径）
//
// 指标：recall@10（对官方 ground_truth_{em,r}.ivecs 前 10 个有效 id）、QPS。
// 运行：./test/garden_realdata_bench_test

#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "roaring/roaring.hh"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class RealDataBenchEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const g_realdata_env = ::testing::AddGlobalTestEnvironment(new RealDataBenchEnv);

static const std::string kDataDir = "/data/workspace/arxiv-for-fanns-medium";
static const int K_TOP = 10;
static const int EF_SEARCH = 50;

using Clock = std::chrono::high_resolution_clock;

// ---------------- fvecs / ivecs 读取 ----------------

static std::vector<float> ReadFvecs(const std::string& path, int& out_dim, int& out_n) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return {};
    }
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), 4);
    in.seekg(0, std::ios::end);
    int64_t file_size = in.tellg();
    int64_t row_bytes = 4 + static_cast<int64_t>(dim) * 4;
    int n = static_cast<int>(file_size / row_bytes);
    in.seekg(0, std::ios::beg);

    std::vector<float> data(static_cast<size_t>(n) * dim);
    std::vector<char> row(static_cast<size_t>(row_bytes));
    for (int i = 0; i < n; ++i) {
        in.read(row.data(), row_bytes);
        std::memcpy(data.data() + static_cast<size_t>(i) * dim, row.data() + 4,
                    static_cast<size_t>(dim) * 4);
    }
    out_dim = dim;
    out_n = n;
    return data;
}

static std::vector<std::vector<int64_t>> ReadIvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::vector<int64_t>> rows;
    while (true) {
        int32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), 4);
        if (!in) break;
        std::vector<int32_t> buf(dim);
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<size_t>(dim) * 4);
        std::vector<int64_t> row;
        row.reserve(dim);
        for (int32_t v : buf) {
            if (v >= 0) row.push_back(v);  // 过滤 -1 填充
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// ---------------- jsonl 属性提取（只取需要的 int 字段，不引入 json 库） ----------------

static bool ExtractIntField(const std::string& line, const char* key, int64_t& out) {
    std::string pat = std::string("\"") + key + "\": ";
    size_t pos = line.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    out = std::strtoll(line.c_str() + pos, nullptr, 10);
    return true;
}

struct DbAttr {
    int64_t num_sub_categories = 0;
    int64_t update_date = 0;
};

static std::vector<DbAttr> ReadDbAttrs(const std::string& path) {
    std::ifstream in(path);
    std::vector<DbAttr> attrs;
    std::string line;
    while (std::getline(in, line)) {
        DbAttr a;
        ExtractIntField(line, "number_of_sub_categories", a.num_sub_categories);
        ExtractIntField(line, "update_date", a.update_date);
        attrs.push_back(a);
    }
    return attrs;
}

static std::vector<int64_t> ReadEmLabels(const std::string& path) {
    std::ifstream in(path);
    std::vector<int64_t> labels;
    std::string line;
    while (std::getline(in, line)) {
        int64_t v = 0;
        if (ExtractIntField(line, "label", v)) labels.push_back(v);
    }
    return labels;
}

struct RangeQuery {
    int64_t lo = 0, hi = 0;
};

static std::vector<RangeQuery> ReadRangeQueries(const std::string& path) {
    std::ifstream in(path);
    std::vector<RangeQuery> qs;
    std::string line;
    while (std::getline(in, line)) {
        RangeQuery q;
        ExtractIntField(line, "range_start", q.lo);
        ExtractIntField(line, "range_end", q.hi);
        qs.push_back(q);
    }
    return qs;
}

// ---------------- 指标 ----------------

static double RecallAtK(const std::vector<int64_t>& got, const std::vector<int64_t>& gt_k) {
    if (gt_k.empty()) return 0.0;
    std::set<int64_t> gt_set(gt_k.begin(), gt_k.end());
    int hit = 0;
    for (int64_t id : got) {
        if (id >= 0 && gt_set.count(id)) ++hit;
    }
    return static_cast<double>(hit) / gt_k.size();
}

struct AggStat {
    double recall_sum = 0.0;
    double total_ms = 0.0;
    int n = 0;
    void Add(double ms, double recall) {
        total_ms += ms;
        recall_sum += recall;
        ++n;
    }
    double AvgRecall() const { return n ? recall_sum / n : 0.0; }
    double Qps() const { return total_ms > 0 ? n / (total_ms / 1000.0) : 0.0; }
};

static std::vector<int64_t> GtTopK(const std::vector<int64_t>& gt_row) {
    std::vector<int64_t> out;
    for (int64_t id : gt_row) {
        if (out.size() >= static_cast<size_t>(K_TOP)) break;
        out.push_back(id);
    }
    return out;
}

// 暴力：遍历位图候选，逐个算 L2，取 top-k（精确，recall 应=1.0）
static std::vector<int64_t> BruteForce(const float* query, const std::vector<float>& db_flat,
                                       int dim, const roaring::Roaring& bm) {
    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve(bm.cardinality());
    for (uint32_t id : bm) {
        const float* vec = db_flat.data() + static_cast<size_t>(id) * dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = query[d] - vec[d];
            dist += diff * diff;
        }
        scored.emplace_back(dist, static_cast<int64_t>(id));
    }
    size_t kk = std::min<size_t>(K_TOP, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + kk, scored.end());
    std::vector<int64_t> ids;
    ids.reserve(kk);
    for (size_t i = 0; i < kk; ++i) ids.push_back(scored[i].second);
    return ids;
}

TEST(RealDataBench, EmAndRange) {
    int dim = 0, n_db = 0, n_q = 0, qdim = 0;
    std::printf("[load] database_vectors.fvecs ...\n");
    auto db_flat = ReadFvecs(kDataDir + "/database_vectors.fvecs", dim, n_db);
    std::printf("[load] db: n=%d dim=%d\n", n_db, dim);
    auto q_flat = ReadFvecs(kDataDir + "/query_vectors.fvecs", qdim, n_q);
    std::printf("[load] query: n=%d dim=%d\n", n_q, qdim);
    auto attrs = ReadDbAttrs(kDataDir + "/database_attributes.jsonl");
    auto em_labels = ReadEmLabels(kDataDir + "/em_query_attributes.jsonl");
    auto r_queries = ReadRangeQueries(kDataDir + "/r_query_attributes.jsonl");
    auto gt_em = ReadIvecs(kDataDir + "/ground_truth_em.ivecs");
    auto gt_r = ReadIvecs(kDataDir + "/ground_truth_r.ivecs");
    ASSERT_EQ(n_db, 100000);
    ASSERT_EQ(static_cast<int>(attrs.size()), n_db);

    auto get_vec = [&](int i) {
        return std::vector<float>(db_flat.begin() + static_cast<size_t>(i) * dim,
                                  db_flat.begin() + static_cast<size_t>(i + 1) * dim);
    };
    auto get_query = [&](int i) {
        return std::vector<float>(q_flat.begin() + static_cast<size_t>(i) * qdim,
                                  q_flat.begin() + static_cast<size_t>(i + 1) * qdim);
    };

    // em 各 label 的等值位图
    std::map<int64_t, roaring::Roaring> em_bitmaps;
    for (int i = 0; i < n_db; ++i) em_bitmaps[attrs[i].num_sub_categories].add(static_cast<uint32_t>(i));

    // update_date 排序索引，用于 r 查询快速建位图
    std::vector<std::pair<int64_t, int>> sorted_dates;
    sorted_dates.reserve(n_db);
    for (int i = 0; i < n_db; ++i) sorted_dates.emplace_back(attrs[i].update_date, i);
    std::sort(sorted_dates.begin(), sorted_dates.end());
    auto build_range_bitmap = [&](int64_t lo, int64_t hi) {
        roaring::Roaring bm;
        auto it_lo = std::lower_bound(sorted_dates.begin(), sorted_dates.end(), std::make_pair(lo, 0));
        auto it_hi = std::upper_bound(sorted_dates.begin(), sorted_dates.end(), std::make_pair(hi, n_db));
        for (auto it = it_lo; it != it_hi; ++it) bm.add(static_cast<uint32_t>(it->second));
        return bm;
    };

    // 构建 HNSW 全量图
    std::printf("[build] HNSW full graph (M=16, ef_construction=200) ...\n");
    auto th0 = Clock::now();
    HNSWLibIndex hnsw(dim, n_db, IndexFactory::MetricType::L2, 16, 200);
    for (int i = 0; i < n_db; ++i) hnsw.InsertVectors(get_vec(i), i);
    std::printf("[build] HNSW done: %.1fs\n", std::chrono::duration<double>(Clock::now() - th0).count());

    // 构建 GARDEN（离散 subcat 懒建子图 + 连续 update_date 分桶）
    // update_date 实测 [13655,20216]，bucket=1000 -> 8 桶；r 查询 range 宽 p50≈3277，覆盖数桶。
    std::printf("[build] GARDEN (discrete=subcat; continuous=update_date [13655,20216] bucket=1000) ...\n");
    auto tg0 = Clock::now();
    GardenIndex garden(dim, n_db, IndexFactory::MetricType::L2);
    garden.RegisterDiscreteField("subcat");
    garden.RegisterContinuousField("update_date", 13655, 20216, 1000);
    for (int i = 0; i < n_db; ++i) {
        std::map<std::string, int64_t> int_fields = {{"update_date", attrs[i].update_date}};
        std::map<std::string, std::string> str_fields = {{"subcat", std::to_string(attrs[i].num_sub_categories)}};
        garden.Insert(get_vec(i), i, int_fields, str_fields);
    }
    std::printf("[build] GARDEN done: %.1fs\n", std::chrono::duration<double>(Clock::now() - tg0).count());

    // ==================== EM ====================
    std::printf("\n==== EM（number_of_sub_categories==label）recall@%d ====\n", K_TOP);
    std::printf("| label | 选择性 | 候选 | Q数 | 暴力召回 | GARDEN召回 | HNSW中滤召回 | 暴力QPS | GARDEN QPS | HNSW中滤QPS | GARDEN路径 |\n");
    std::printf("|-------|--------|------|-----|----------|------------|--------------|---------|------------|-------------|------------|\n");

    std::map<int64_t, std::vector<int>> em_by_label;
    for (int i = 0; i < static_cast<int>(em_labels.size()); ++i) em_by_label[em_labels[i]].push_back(i);

    const int kMaxQ = 200;
    for (const auto& kv : em_by_label) {
        const int64_t label = kv.first;
        const std::vector<int>& qids = kv.second;
        roaring::Roaring& bm = em_bitmaps[label];
        uint64_t candidate = bm.cardinality();
        double sel = static_cast<double>(candidate) / n_db;
        int take = std::min<int>(kMaxQ, qids.size());

        AggStat brute, garden_s, hnsw_s;
        for (int j = 0; j < take; ++j) {
            int qi = qids[j];
            auto query = get_query(qi);
            auto gt_k = GtTopK(gt_em[qi]);

            auto t0 = Clock::now();
            auto b_ids = BruteForce(query.data(), db_flat, dim, bm);
            auto t1 = Clock::now();
            brute.Add(std::chrono::duration<double, std::milli>(t1 - t0).count(), RecallAtK(b_ids, gt_k));

            GardenFilter gf;
            gf.field = "subcat";
            gf.discrete_values = {std::to_string(label)};
            gf.bitmap = &bm.roaring;
            gf.cardinality = candidate;
            auto t2 = Clock::now();
            auto [g_ids, g_d] = garden.Search(query, K_TOP, {gf}, nullptr, EF_SEARCH);
            auto t3 = Clock::now();
            garden_s.Add(std::chrono::duration<double, std::milli>(t3 - t2).count(), RecallAtK(g_ids, gt_k));

            auto t4 = Clock::now();
            auto [h_ids, h_d] = hnsw.SearchVectors(query, K_TOP, &bm.roaring, EF_SEARCH);
            auto t5 = Clock::now();
            hnsw_s.Add(std::chrono::duration<double, std::milli>(t5 - t4).count(), RecallAtK(h_ids, gt_k));
        }
        const char* path = (candidate < 10000) ? "暴力" : (sel > 0.8 ? "全量图" : "子图");
        std::printf("| %5lld | %5.1f%% | %6llu | %3d | %8.3f | %10.3f | %12.3f | %7.0f | %10.0f | %11.0f | %s |\n",
                    (long long)label, sel * 100, (unsigned long long)candidate, take,
                    brute.AvgRecall(), garden_s.AvgRecall(), hnsw_s.AvgRecall(),
                    brute.Qps(), garden_s.Qps(), hnsw_s.Qps(), path);
    }

    // ==================== R ====================
    std::printf("\n==== R（range_start<=update_date<=range_end）recall@%d ====\n", K_TOP);
    struct RGroup { const char* name; int64_t lo_c, hi_c; int max_q; };
    std::vector<RGroup> groups = {
        {"<10k(暴力)", 0, 10000, 100},
        {"10k-80k(子图)", 10000, 80000, 300},
        {">80k(全量图)", 80000, INT64_MAX, 100},
    };
    std::printf("| 分组 | Q数 | 暴力召回 | GARDEN召回 | HNSW中滤召回 | 暴力QPS | GARDEN QPS | HNSW中滤QPS |\n");
    std::printf("|------|-----|----------|------------|--------------|---------|------------|-------------|\n");

    for (const auto& g : groups) {
        AggStat brute, garden_s, hnsw_s;
        int taken = 0;
        for (int qi = 0; qi < static_cast<int>(r_queries.size()) && taken < g.max_q; ++qi) {
            auto& rq = r_queries[qi];
            roaring::Roaring bm = build_range_bitmap(rq.lo, rq.hi);
            uint64_t candidate = bm.cardinality();
            if (candidate < static_cast<uint64_t>(g.lo_c) || candidate >= static_cast<uint64_t>(g.hi_c)) continue;
            ++taken;

            auto query = get_query(qi);
            auto gt_k = GtTopK(gt_r[qi]);

            auto t0 = Clock::now();
            auto b_ids = BruteForce(query.data(), db_flat, dim, bm);
            auto t1 = Clock::now();
            brute.Add(std::chrono::duration<double, std::milli>(t1 - t0).count(), RecallAtK(b_ids, gt_k));

            GardenFilter gf;
            gf.field = "update_date";
            gf.has_range = true;
            gf.range_min = rq.lo;
            gf.range_max = rq.hi;
            gf.bitmap = &bm.roaring;
            gf.cardinality = candidate;
            auto t2 = Clock::now();
            auto [g_ids, g_d] = garden.Search(query, K_TOP, {gf}, nullptr, EF_SEARCH);
            auto t3 = Clock::now();
            garden_s.Add(std::chrono::duration<double, std::milli>(t3 - t2).count(), RecallAtK(g_ids, gt_k));

            auto t4 = Clock::now();
            auto [h_ids, h_d] = hnsw.SearchVectors(query, K_TOP, &bm.roaring, EF_SEARCH);
            auto t5 = Clock::now();
            hnsw_s.Add(std::chrono::duration<double, std::milli>(t5 - t4).count(), RecallAtK(h_ids, gt_k));
        }
        std::printf("| %s | %3d | %8.3f | %10.3f | %12.3f | %7.0f | %10.0f | %11.0f |\n",
                    g.name, taken, brute.AvgRecall(), garden_s.AvgRecall(), hnsw_s.AvgRecall(),
                    brute.Qps(), garden_s.Qps(), hnsw_s.Qps());
    }

    std::printf("\n[done] EM+R 三路对比完成。recall 对官方 GT@%d，HNSW 参数 M=16/efc=200/efs=%d。\n",
                K_TOP, EF_SEARCH);
}

}  // namespace vectordb
