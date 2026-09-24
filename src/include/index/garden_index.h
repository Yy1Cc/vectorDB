#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <cstdint>
#include <utility>
#include "index/hnswlib_index.h"
#include "index/index_factory.h"
#include "roaring/roaring.h"

namespace vectordb {

// GARDEN 标量过滤条件（由外部 search 请求解析后传入）
// 每个条件对应一个 AND 子句，如 brand IN [apple, xiaomi] 或 1000 <= price <= 5000
struct GardenFilter {
    std::string field;

    // 离散条件：field IN [discrete_values]
    std::vector<std::string> discrete_values;

    // 连续条件：range_min <= field <= range_max
    bool has_range = false;
    int64_t range_min = 0;
    int64_t range_max = 0;

    // 外部预计算的 bitmap（用于没有子图的 Grafter 条件）
    // Pruner 条件不需要 bitmap（直接用子图搜索）
    const roaring_bitmap_t* bitmap = nullptr;
    size_t cardinality = 0;  // bitmap 基数，用于选择性评估
};

// GardenIndex：GARDEN 标量过滤向量检索引擎
//
// 参考 SimOL GARDEN 论文实现三个核心模块：
// 1. 多子图：离散标签子图（每标签值独立 HNSW）+ 连续分桶子图（按数值范围分桶）
// 2. 多策略路由：候选集 < brute_bound 暴力检索 / 高选择性全量图 / 低选择性子图搜索
// 3. Pruner/Grafter/Selector 分级：按选择性把多 AND 条件拆为三段式处理
//
// 设计要点：
// - 全量图始终维护，无过滤条件时走标准 HNSW 搜索
// - 离散子图为每个标签值建独立 HNSWLibIndex，多入口并行搜索后合并
// - 连续字段按 bucket_size 分桶，每桶独立 HNSWLibIndex
// - 暴力检索维护 id_to_vector_ 原始向量表，候选集小时直接算距离
// - 不修改 hnswlib 内部，纯外层多实例组合
class GardenIndex {
public:
    GardenIndex(int dim, int num_data,
                IndexFactory::MetricType metric = IndexFactory::MetricType::L2);
    ~GardenIndex();

    // 禁止拷贝（持有 HNSW 图资源）
    GardenIndex(const GardenIndex&) = delete;
    auto operator=(const GardenIndex&) -> GardenIndex& = delete;

    // ========== 子图注册 ==========

    // 注册离散过滤字段（如 brand），插入时自动为每个值建子图
    void RegisterDiscreteField(const std::string& field);

    // 注册连续过滤字段（如 price），按数值范围分桶建子图
    void RegisterContinuousField(const std::string& field,
                                 int64_t min_val, int64_t max_val,
                                 int bucket_size = kBruteBound);

    // 查询某字段是否已注册离散子图
    auto HasDiscreteSubgraph(const std::string& field) const -> bool;

    // 查询某字段是否已注册连续子图
    auto HasContinuousSubgraph(const std::string& field) const -> bool;

    // ========== 插入 ==========

    // 插入单条向量
    // int_fields:  整型标量字段（如 price=3000），用于连续分桶路由
    // str_fields:  字符串标量字段（如 brand=apple），用于离散子图路由
    void Insert(const std::vector<float>& vec, int64_t id,
                const std::map<std::string, int64_t>& int_fields,
                const std::map<std::string, std::string>& str_fields);

    // 按 ID 批量删除向量：全量图 mark delete + 所有关联子图 mark delete
    // + id_to_vector_ 清除 + 计数更新 + 反向映射清除。
    // 内部用 try-catch 容错（HNSWLibIndex::RemoveVectors 对不存在 label 抛异常）。
    void RemoveVectors(const std::vector<int64_t>& ids);

    // ========== 搜索 ==========

    // GARDEN 搜索：核心路由 + Pruner/Grafter/Selector 三段式
    // filters:         标量过滤条件列表（已带 bitmap + cardinality）
    // selector_bitmap: 黑名单等 Selector 条件（结果中排除这些 ID）
    auto Search(const std::vector<float>& query, int k,
                const std::vector<GardenFilter>& filters,
                const roaring_bitmap_t* selector_bitmap = nullptr,
                int ef_search = 50)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    // 无过滤搜索（走全量图标准 HNSW）
    auto Search(const std::vector<float>& query, int k, int ef_search = 50)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    // ========== 持久化 ==========

    void SaveIndex(const std::string& path);
    void LoadIndex(const std::string& path);

    // ========== 状态查询 ==========

    auto GetTotalCount() const -> size_t { return total_count_; }
    auto GetDim() const -> int { return dim_; }

    // 获取离散子图的标签值列表（用于测试和调试）
    auto GetDiscreteValues(const std::string& field) const -> std::vector<std::string>;

    // GARDEN 论文参数
    static constexpr int kBruteBound = 10000;        // 暴力检索阈值（绝对上限）
    static constexpr double kFullSearchRate = 0.8;    // > 此值：全量图 + Selector 后过滤
    // 暴力阈值的相对部分：候选集低于本地总量该比例才考虑走暴力。
    // 必要性：分区部署下每个分区只持有 1/N 数据，候选集被摊薄到 1/N。
    // 若阈值只看绝对量，分区越多越容易掉进暴力路径 —— 而实测（10万×4096维）
    // 暴力路径 QPS 仅 1~11，子图路径 171~193，差距达两个数量级。
    static constexpr double kBruteForceRate = 0.01;
    static constexpr int kMinBruteBound = 500;       // 阈值下限，避免极小数据集反而绕远路

private:
    int dim_;
    int num_data_;
    IndexFactory::MetricType metric_;
    size_t total_count_ = 0;

    // 全量 HNSW 图（无过滤时用）
    std::unique_ptr<HNSWLibIndex> full_index_;

    // 原始向量表（暴力检索用）
    std::map<int64_t, std::vector<float>> id_to_vector_;

    // 反向映射：id -> (field -> value)，用于删除时精确定位离散子图
    std::map<int64_t, std::map<std::string, std::string>> id_to_discrete_;
    // 反向映射：id -> (field -> bucket_idx)，用于删除时精确定位连续分桶子图
    std::map<int64_t, std::map<std::string, int>> id_to_continuous_bucket_;

    // 离散子图：field -> (value -> HNSWLibIndex)
    std::map<std::string, std::map<std::string, std::unique_ptr<HNSWLibIndex>>> discrete_subgraphs_;
    std::map<std::string, std::map<std::string, size_t>> discrete_counts_;  // 子图元素计数
    std::set<std::string> discrete_fields_;

    // 连续分桶子图
    struct ContinuousBucket {
        int64_t bucket_min;
        int64_t bucket_max;
        std::unique_ptr<HNSWLibIndex> index;
        size_t count = 0;
    };
    struct ContinuousFieldConfig {
        int64_t field_min = 0;
        int64_t field_max = 0;
        int bucket_size = kBruteBound;
        std::vector<ContinuousBucket> buckets;  // 按 bucket_min 排序
    };
    std::map<std::string, ContinuousFieldConfig> continuous_fields_;

    // ========== 内部方法 ==========

    // 创建新子图（统一容量管理）
    auto CreateSubgraph(size_t expected_size) -> std::unique_ptr<HNSWLibIndex>;

    // 获取或创建离散子图
    auto GetOrCreateDiscreteSubgraph(const std::string& field, const std::string& value)
        -> HNSWLibIndex*;

    // 定位连续字段值对应的桶索引（返回 -1 表示未找到）
    auto FindBucket(const std::string& field, int64_t value) const -> int;

    // 暴力检索：在 candidate_bitmap 候选集上直接算距离
    auto BruteForceSearch(const std::vector<float>& query, int k,
                          const roaring_bitmap_t* candidate_bitmap,
                          const roaring_bitmap_t* selector_bitmap)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    // 多入口子图搜索：并行搜索多个子图，合并 top-k
    auto MultiEntrySearch(const std::vector<float>& query, int k,
                          const std::vector<HNSWLibIndex*>& subgraphs,
                          const roaring_bitmap_t* grafter_bitmap,
                          const roaring_bitmap_t* selector_bitmap,
                          int ef_search)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    // 评估选择性（cardinality / total_count_）
    auto EstimateSelectivity(size_t cardinality) const -> double;
};

}  // namespace vectordb
