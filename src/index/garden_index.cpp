#include "index/garden_index.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "logger/logger.h"

namespace vectordb {

// ============================================================================
// 构造 / 析构
// ============================================================================

GardenIndex::GardenIndex(int dim, int num_data, IndexFactory::MetricType metric)
    : dim_(dim), num_data_(num_data), metric_(metric) {
    // GARDEN 基于 HNSW，目前 HNSWLibIndex 仅支持 L2
    full_index_ = std::make_unique<HNSWLibIndex>(dim, num_data, metric, 16, 200);
    global_logger->info("GardenIndex created: dim={}, num_data={}, metric={}",
                        dim, num_data, static_cast<int>(metric));
}

GardenIndex::~GardenIndex() = default;

// ============================================================================
// 子图注册
// ============================================================================

void GardenIndex::RegisterDiscreteField(const std::string& field) {
    discrete_fields_.insert(field);
    discrete_subgraphs_[field];  // 默认构造空 map（不用 = {} 避免 initializer_list 拷贝）
    discrete_counts_[field];
    global_logger->info("GardenIndex: registered discrete field '{}'", field);
}

void GardenIndex::RegisterContinuousField(const std::string& field,
                                          int64_t min_val, int64_t max_val,
                                          int bucket_size) {
    if (bucket_size <= 0) bucket_size = kBruteBound;
    if (min_val >= max_val) {
        global_logger->error("GardenIndex: invalid range for continuous field '{}': [{}, {}]",
                             field, min_val, max_val);
        return;
    }

    ContinuousFieldConfig config;
    config.field_min = min_val;
    config.field_max = max_val;
    config.bucket_size = bucket_size;

    // 预分配桶空间，避免 vector 扩容触发 unique_ptr 拷贝
    size_t num_buckets = static_cast<size_t>((max_val - min_val) / bucket_size) + 1;
    config.buckets.reserve(num_buckets);

    // 按 bucket_size 切分 [min_val, max_val]
    for (int64_t start = min_val; start < max_val; start += bucket_size) {
        int64_t end = std::min(start + bucket_size - 1, max_val);
        ContinuousBucket bucket;
        bucket.bucket_min = start;
        bucket.bucket_max = end;
        // 子图初始容量设为 bucket_size（BatchInsert 会自动 resize）
        bucket.index = CreateSubgraph(static_cast<size_t>(bucket_size));
        config.buckets.push_back(std::move(bucket));
    }

    continuous_fields_[field] = std::move(config);
    global_logger->info("GardenIndex: registered continuous field '{}' range=[{}, {}], {} buckets",
                        field, min_val, max_val, continuous_fields_[field].buckets.size());
}

auto GardenIndex::HasDiscreteSubgraph(const std::string& field) const -> bool {
    return discrete_fields_.count(field) > 0;
}

auto GardenIndex::HasContinuousSubgraph(const std::string& field) const -> bool {
    return continuous_fields_.count(field) > 0;
}

auto GardenIndex::GetDiscreteValues(const std::string& field) const -> std::vector<std::string> {
    std::vector<std::string> values;
    auto it = discrete_counts_.find(field);
    if (it != discrete_counts_.end()) {
        for (const auto& [v, cnt] : it->second) {
            values.push_back(v);
        }
    }
    return values;
}

// ============================================================================
// 内部：子图创建与定位
// ============================================================================

auto GardenIndex::CreateSubgraph(size_t expected_size) -> std::unique_ptr<HNSWLibIndex> {
    // 初始容量至少 1024，避免频繁 resize
    size_t capacity = std::max(expected_size, static_cast<size_t>(1024));
    return std::make_unique<HNSWLibIndex>(dim_, static_cast<int>(capacity), metric_, 16, 200);
}

auto GardenIndex::GetOrCreateDiscreteSubgraph(const std::string& field,
                                               const std::string& value) -> HNSWLibIndex* {
    auto& value_map = discrete_subgraphs_[field];
    auto it = value_map.find(value);
    if (it == value_map.end()) {
        // 新标签值，创建子图（初始容量 1024，BatchInsert 会自动扩容）
        auto sub = CreateSubgraph(1024);
        HNSWLibIndex* raw = sub.get();
        value_map[value] = std::move(sub);
        discrete_counts_[field][value] = 0;
        return raw;
    }
    return it->second.get();
}

auto GardenIndex::FindBucket(const std::string& field, int64_t value) const -> int {
    auto it = continuous_fields_.find(field);
    if (it == continuous_fields_.end()) return -1;

    const auto& config = it->second;
    if (value < config.field_min || value > config.field_max) return -1;

    // 二分查找：buckets 按 bucket_min 升序
    int lo = 0, hi = static_cast<int>(config.buckets.size()) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const auto& b = config.buckets[mid];
        if (value < b.bucket_min) {
            hi = mid - 1;
        } else if (value > b.bucket_max) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

// ============================================================================
// 插入
// ============================================================================

void GardenIndex::Insert(const std::vector<float>& vec, int64_t id,
                         const std::map<std::string, int64_t>& int_fields,
                         const std::map<std::string, std::string>& str_fields) {
    // 1. 全量图（用 BatchInsert 确保自动 resize）
    full_index_->BatchInsertVectors(vec, 1, {id});

    // 自动注册新的离散字段（字符串字段未注册时自动建子图）
    for (const auto& [field, value] : str_fields) {
        if (discrete_fields_.find(field) == discrete_fields_.end()) {
            RegisterDiscreteField(field);
        }
    }

    // 2. 离散子图
    for (const auto& field : discrete_fields_) {
        auto it = str_fields.find(field);
        if (it != str_fields.end()) {
            HNSWLibIndex* sub = GetOrCreateDiscreteSubgraph(field, it->second);
            sub->BatchInsertVectors(vec, 1, {id});
            discrete_counts_[field][it->second]++;
            // 反向映射：删除时据此精确定位子图
            id_to_discrete_[id][field] = it->second;
        }
    }

    // 3. 连续分桶子图
    for (auto& [field, config] : continuous_fields_) {
        auto it = int_fields.find(field);
        if (it != int_fields.end()) {
            int bucket_idx = FindBucket(field, it->second);
            if (bucket_idx >= 0) {
                auto& bucket = config.buckets[bucket_idx];
                bucket.index->BatchInsertVectors(vec, 1, {id});
                bucket.count++;
                // 反向映射：删除时据此精确定位分桶子图
                id_to_continuous_bucket_[id][field] = bucket_idx;
            }
        }
    }

    // 4. 原始向量表（暴力检索用）
    id_to_vector_[id] = vec;
    total_count_++;
}

// ============================================================================
// 删除
// ============================================================================

void GardenIndex::RemoveVectors(const std::vector<int64_t>& ids) {
    for (int64_t id : ids) {
        // 1. 全量图 mark delete（HNSWLibIndex::RemoveVectors 对不存在 label 抛
        //    std::runtime_error，这里容错：upsert 先删旧时 id 可能已不在图内）
        try {
            full_index_->RemoveVectors({id});
        } catch (const std::runtime_error& e) {
            global_logger->debug("GardenIndex::RemoveVectors: id {} not in full graph: {}",
                                 id, e.what());
        }

        // 2. 离散子图：查反向映射精确定位，逐个子图 mark delete
        auto disc_it = id_to_discrete_.find(id);
        if (disc_it != id_to_discrete_.end()) {
            for (const auto& [field, value] : disc_it->second) {
                auto fIt = discrete_subgraphs_.find(field);
                if (fIt != discrete_subgraphs_.end()) {
                    auto vIt = fIt->second.find(value);
                    if (vIt != fIt->second.end() && vIt->second) {
                        try {
                            vIt->second->RemoveVectors({id});
                        } catch (const std::runtime_error&) {
                            // id 不在该子图，忽略
                        }
                        auto cntIt = discrete_counts_.find(field);
                        if (cntIt != discrete_counts_.end()) {
                            auto cvIt = cntIt->second.find(value);
                            if (cvIt != cntIt->second.end() && cvIt->second > 0) {
                                cvIt->second--;
                            }
                        }
                    }
                }
            }
            id_to_discrete_.erase(disc_it);
        }

        // 3. 连续分桶子图：查反向映射精确定位桶，mark delete
        auto cont_it = id_to_continuous_bucket_.find(id);
        if (cont_it != id_to_continuous_bucket_.end()) {
            for (const auto& [field, bucket_idx] : cont_it->second) {
                auto fIt = continuous_fields_.find(field);
                if (fIt != continuous_fields_.end() && bucket_idx >= 0 &&
                    static_cast<size_t>(bucket_idx) < fIt->second.buckets.size()) {
                    auto& bucket = fIt->second.buckets[bucket_idx];
                    if (bucket.index) {
                        try {
                            bucket.index->RemoveVectors({id});
                        } catch (const std::runtime_error&) {
                            // id 不在该桶，忽略
                        }
                        if (bucket.count > 0) bucket.count--;
                    }
                }
            }
            id_to_continuous_bucket_.erase(cont_it);
        }

        // 4. 原始向量表 + 总计数
        id_to_vector_.erase(id);
        if (total_count_ > 0) total_count_--;
    }

    global_logger->debug("GardenIndex::RemoveVectors: processed {} ids", ids.size());
}

// ============================================================================
// 搜索：路由决策 + Pruner/Grafter/Selector
// ============================================================================

auto GardenIndex::Search(const std::vector<float>& query, int k,
                         const std::vector<GardenFilter>& filters,
                         const roaring_bitmap_t* selector_bitmap,
                         int ef_search) -> std::pair<std::vector<int64_t>, std::vector<float>> {
    // ---- 无过滤条件：走全量图标准 HNSW ----
    if (filters.empty()) {
        global_logger->debug("GardenIndex: no filter, full graph search");
        return full_index_->SearchVectors(query, k, nullptr, ef_search);
    }

    // ---- 评估各条件选择性，分级为 Pruner / Grafter / Selector ----
    // Pruner: 选择性最低且"有离散子图"的条件 → 用子图搜索剪枝
    // Grafter: 其余有 bitmap 的条件 → AND 合并为 grafter_bitmap
    // Selector: selector_bitmap 参数 → 后过滤不入结果

    std::string pruner_field;
    std::vector<std::string> pruner_values;
    bool has_pruner = false;
    double pruner_selectivity = 2.0;  // 初始化为 >1

    // 连续条件也可以做 Pruner（用分桶子图）
    bool pruner_is_continuous = false;
    int64_t pruner_range_min = 0, pruner_range_max = 0;

    // 合并 Grafter bitmap
    roaring_bitmap_t* grafter_bitmap = nullptr;

    for (const auto& filter : filters) {
        bool has_disc_sub = HasDiscreteSubgraph(filter.field);
        bool has_cont_sub = HasContinuousSubgraph(filter.field);

        if ((has_disc_sub || has_cont_sub) && filter.cardinality > 0) {
            // 有子图的条件：候选 Pruner
            double sel = EstimateSelectivity(filter.cardinality);
            if (sel < pruner_selectivity) {
                pruner_selectivity = sel;
                pruner_field = filter.field;
                has_pruner = true;
                pruner_is_continuous = has_cont_sub && !has_disc_sub;
                if (has_disc_sub) {
                    pruner_values = filter.discrete_values;
                }
                if (has_cont_sub) {
                    pruner_range_min = filter.range_min;
                    pruner_range_max = filter.range_max;
                }
            }
        }

        // 所有条件都合并到 grafter_bitmap（Pruner 条件也并入，用于子图内二次过滤）
        if (filter.bitmap != nullptr) {
            if (grafter_bitmap == nullptr) {
                grafter_bitmap = roaring_bitmap_copy(filter.bitmap);
            } else {
                roaring_bitmap_and_inplace(grafter_bitmap, filter.bitmap);
            }
        }
    }

    // ---- 计算总候选集大小 ----
    size_t candidate_count = 0;
    if (grafter_bitmap != nullptr) {
        candidate_count = roaring_bitmap_get_cardinality(grafter_bitmap);
    } else if (has_pruner) {
        // 没有 grafter bitmap 但有 Pruner，用 Pruner 子图大小估算
        if (!pruner_is_continuous) {
            for (const auto& v : pruner_values) {
                auto it = discrete_counts_.find(pruner_field);
                if (it != discrete_counts_.end()) {
                    auto vit = it->second.find(v);
                    if (vit != it->second.end()) {
                        candidate_count += vit->second;
                    }
                }
            }
        } else {
            // 连续 Pruner：累加覆盖桶的 count
            auto it = continuous_fields_.find(pruner_field);
            if (it != continuous_fields_.end()) {
                for (const auto& bucket : it->second.buckets) {
                    if (bucket.bucket_max >= pruner_range_min && bucket.bucket_min <= pruner_range_max) {
                        candidate_count += bucket.count;
                    }
                }
            }
        }
    }

    double overall_selectivity = (total_count_ > 0)
        ? static_cast<double>(candidate_count) / static_cast<double>(total_count_)
        : 1.0;

    global_logger->info("GardenIndex Search: total={}, candidate={}, selectivity={:.4f}, has_pruner={}",
                        total_count_, candidate_count, overall_selectivity, has_pruner);

    // ---- 路由决策 ----

    // 策略 1: 候选集 < brute_bound → 暴力检索
    if (candidate_count > 0 && candidate_count < static_cast<size_t>(kBruteBound)) {
        global_logger->debug("GardenIndex: brute force (candidate={} < {})", candidate_count, kBruteBound);
        auto result = BruteForceSearch(query, k, grafter_bitmap, selector_bitmap);
        if (grafter_bitmap) roaring_bitmap_free(grafter_bitmap);
        return result;
    }

    // 策略 2: 选择性很高 (> kFullSearchRate) → 全量图 + bitmap IDSelector
    if (overall_selectivity > kFullSearchRate || !has_pruner) {
        global_logger->debug("GardenIndex: full graph + bitmap (selectivity={:.4f})", overall_selectivity);
        // 合并 grafter + selector 为最终过滤 bitmap
        roaring_bitmap_t* combined = nullptr;
        if (grafter_bitmap != nullptr) {
            combined = roaring_bitmap_copy(grafter_bitmap);
            if (selector_bitmap != nullptr) {
                roaring_bitmap_and_inplace(combined, selector_bitmap);
            }
        } else if (selector_bitmap != nullptr) {
            combined = roaring_bitmap_copy(selector_bitmap);
        }
        // 增大 ef 补偿低选择性（ACORN 退化）
        int adjusted_ef = std::max(ef_search, static_cast<int>(k * 10));
        auto result = full_index_->SearchVectors(query, k, combined, adjusted_ef);
        if (combined) roaring_bitmap_free(combined);
        if (grafter_bitmap) roaring_bitmap_free(grafter_bitmap);
        return result;
    }

    // 策略 3: 有 Pruner 子图 → 多入口子图搜索 + Grafter bitmap 过滤
    std::vector<HNSWLibIndex*> pruner_subgraphs;

    if (!pruner_is_continuous) {
        // 离散 Pruner：收集各标签值的子图
        auto& value_map = discrete_subgraphs_[pruner_field];
        for (const auto& v : pruner_values) {
            auto it = value_map.find(v);
            if (it != value_map.end() && it->second != nullptr) {
                pruner_subgraphs.push_back(it->second.get());
            }
        }
    } else {
        // 连续 Pruner：收集覆盖范围的分桶子图
        auto it = continuous_fields_.find(pruner_field);
        if (it != continuous_fields_.end()) {
            for (const auto& bucket : it->second.buckets) {
                if (bucket.bucket_max >= pruner_range_min && bucket.bucket_min <= pruner_range_max) {
                    if (bucket.index && bucket.count > 0) {
                        pruner_subgraphs.push_back(bucket.index.get());
                    }
                }
            }
        }
    }

    if (pruner_subgraphs.empty()) {
        // Pruner 子图为空，退化为全量图
        global_logger->debug("GardenIndex: pruner subgraphs empty, fallback to full graph");
        int adjusted_ef = std::max(ef_search, static_cast<int>(k * 10));
        auto result = full_index_->SearchVectors(query, k, grafter_bitmap, adjusted_ef);
        if (grafter_bitmap) roaring_bitmap_free(grafter_bitmap);
        return result;
    }

    global_logger->debug("GardenIndex: multi-entry search on {} subgraphs (field='{}')",
                         pruner_subgraphs.size(), pruner_field);
    auto result = MultiEntrySearch(query, k, pruner_subgraphs,
                                   grafter_bitmap, selector_bitmap, ef_search);
    if (grafter_bitmap) roaring_bitmap_free(grafter_bitmap);
    return result;
}

auto GardenIndex::Search(const std::vector<float>& query, int k, int ef_search)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
    return full_index_->SearchVectors(query, k, nullptr, ef_search);
}

// ============================================================================
// 暴力检索
// ============================================================================

auto GardenIndex::BruteForceSearch(const std::vector<float>& query, int k,
                                   const roaring_bitmap_t* candidate_bitmap,
                                   const roaring_bitmap_t* selector_bitmap)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
    // 遍历 candidate_bitmap 中的 ID，算 L2 距离
    std::vector<std::pair<float, int64_t>> scored;

    if (candidate_bitmap != nullptr) {
        roaring_uint32_iterator_t it;
        roaring_init_iterator(candidate_bitmap, &it);
        while (it.has_value) {
            uint32_t id = it.current_value;
            // Selector 过滤：selector_bitmap 中的 ID 排除
            if (selector_bitmap != nullptr && roaring_bitmap_contains(selector_bitmap, id)) {
                roaring_advance_uint32_iterator(&it);
                continue;
            }
            auto vec_it = id_to_vector_.find(static_cast<int64_t>(id));
            if (vec_it != id_to_vector_.end()) {
                float dist = 0.0f;
                const auto& vec = vec_it->second;
                for (int i = 0; i < dim_ && i < static_cast<int>(vec.size()); ++i) {
                    float diff = query[i] - vec[i];
                    dist += diff * diff;
                }
                scored.emplace_back(dist, static_cast<int64_t>(id));
            }
            roaring_advance_uint32_iterator(&it);
        }
    } else {
        // 无 bitmap，遍历所有向量
        for (const auto& [id, vec] : id_to_vector_) {
            if (selector_bitmap != nullptr &&
                roaring_bitmap_contains(selector_bitmap, static_cast<uint32_t>(id))) {
                continue;
            }
            float dist = 0.0f;
            for (int i = 0; i < dim_ && i < static_cast<int>(vec.size()); ++i) {
                float diff = query[i] - vec[i];
                dist += diff * diff;
            }
            scored.emplace_back(dist, id);
        }
    }

    // 取 top-k（距离最小的）
    std::sort(scored.begin(), scored.end());

    std::vector<int64_t> ids(k, -1);
    std::vector<float> dists(k, -1.0f);
    int j = 0;
    for (const auto& [d, id] : scored) {
        if (j >= k) break;
        ids[j] = id;
        dists[j] = d;
        j++;
    }

    global_logger->debug("GardenIndex brute force: evaluated {} candidates, returned {}",
                         scored.size(), j);
    return {ids, dists};
}

// ============================================================================
// 多入口子图搜索
// ============================================================================

auto GardenIndex::MultiEntrySearch(const std::vector<float>& query, int k,
                                   const std::vector<HNSWLibIndex*>& subgraphs,
                                   const roaring_bitmap_t* grafter_bitmap,
                                   const roaring_bitmap_t* selector_bitmap,
                                   int ef_search)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
    // 合并 grafter + selector 为最终过滤 bitmap
    roaring_bitmap_t* combined_filter = nullptr;
    if (grafter_bitmap != nullptr) {
        combined_filter = roaring_bitmap_copy(grafter_bitmap);
        if (selector_bitmap != nullptr) {
            roaring_bitmap_and_inplace(combined_filter, selector_bitmap);
        }
    } else if (selector_bitmap != nullptr) {
        combined_filter = roaring_bitmap_copy(selector_bitmap);
    }

    // 每个 Pruner 子图搜索 top-k，合并到共享候选池
    // 共享候选池用最大堆保留 top-k（按距离从小到大）
    std::vector<std::pair<float, int64_t>> merged;

    for (HNSWLibIndex* sub : subgraphs) {
        // 子图内搜索时用 grafter_bitmap 做 IDSelector
        auto [sub_ids, sub_dists] = sub->SearchVectors(query, k, combined_filter, ef_search);
        for (int i = 0; i < static_cast<int>(sub_ids.size()); ++i) {
            if (sub_ids[i] >= 0) {
                merged.emplace_back(sub_dists[i], sub_ids[i]);
            }
        }
    }

    if (combined_filter) roaring_bitmap_free(combined_filter);

    // 去重（同一 ID 可能在多个子图中出现）+ 取 top-k
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end(),
                             [](const auto& a, const auto& b) { return a.second == b.second; }),
                 merged.end());

    std::vector<int64_t> ids(k, -1);
    std::vector<float> dists(k, -1.0f);
    int j = 0;
    for (const auto& [d, id] : merged) {
        if (j >= k) break;
        ids[j] = id;
        dists[j] = d;
        j++;
    }

    global_logger->debug("GardenIndex multi-entry: searched {} subgraphs, merged {} candidates, returned {}",
                         subgraphs.size(), merged.size(), j);
    return {ids, dists};
}

// ============================================================================
// 选择性评估
// ============================================================================

auto GardenIndex::EstimateSelectivity(size_t cardinality) const -> double {
    if (total_count_ == 0) return 1.0;
    return static_cast<double>(cardinality) / static_cast<double>(total_count_);
}

// ============================================================================
// 持久化
// ============================================================================

void GardenIndex::SaveIndex(const std::string& path) {
    // 全量图
    full_index_->SaveIndex(path + "_full.index");

    // 离散子图
    for (const auto& [field, value_map] : discrete_subgraphs_) {
        for (const auto& [value, index] : value_map) {
            if (index) {
                index->SaveIndex(path + "_disc_" + field + "_" + value + ".index");
            }
        }
    }

    // 连续分桶子图
    for (const auto& [field, config] : continuous_fields_) {
        for (size_t i = 0; i < config.buckets.size(); ++i) {
            if (config.buckets[i].index) {
                config.buckets[i].index->SaveIndex(
                    path + "_cont_" + field + "_" + std::to_string(i) + ".index");
            }
        }
    }

    // 元数据（id_to_vector, counts, config）
    std::ofstream meta(path + "_meta.bin", std::ios::binary);
    if (!meta.is_open()) {
        global_logger->error("GardenIndex: failed to open meta file for saving: {}", path);
        return;
    }
    // 简单文本格式：先写 total_count 和 dim
    meta << total_count_ << " " << dim_ << "\n";
    // id_to_vector
    meta << id_to_vector_.size() << "\n";
    for (const auto& [id, vec] : id_to_vector_) {
        meta << id;
        for (float v : vec) meta << " " << v;
        meta << "\n";
    }
    // 反向映射（REVMAP 哨兵，便于 Load 时识别；假设 field/value 不含空白）
    meta << "REVMAP\n";
    meta << id_to_discrete_.size() << "\n";
    for (const auto& [id, fmap] : id_to_discrete_) {
        meta << id << " " << fmap.size();
        for (const auto& [field, value] : fmap) meta << " " << field << " " << value;
        meta << "\n";
    }
    meta << id_to_continuous_bucket_.size() << "\n";
    for (const auto& [id, fmap] : id_to_continuous_bucket_) {
        meta << id << " " << fmap.size();
        for (const auto& [field, bidx] : fmap) meta << " " << field << " " << bidx;
        meta << "\n";
    }
    // 字段配置（FIELDS 哨兵，Load 时据此重建子图结构，解决重启后子图加载不了的既有限制）
    meta << "FIELDS\n";
    meta << discrete_fields_.size() << "\n";
    for (const auto& field : discrete_fields_) {
        meta << field << "\n";
    }
    meta << continuous_fields_.size() << "\n";
    for (const auto& [field, config] : continuous_fields_) {
        meta << field << " " << config.field_min << " " << config.field_max
             << " " << config.bucket_size << "\n";
    }
    meta.close();
    global_logger->info("GardenIndex: saved to {}", path);
}

void GardenIndex::LoadIndex(const std::string& path) {
    // 全量图
    full_index_->LoadIndex(path + "_full.index");

    // 元数据
    std::ifstream meta(path + "_meta.bin", std::ios::binary);
    if (!meta.is_open()) {
        global_logger->warn("GardenIndex: meta file not found: {}, skipping load", path);
        return;
    }
    meta >> total_count_ >> dim_;
    size_t vec_count;
    meta >> vec_count;
    id_to_vector_.clear();
    for (size_t i = 0; i < vec_count; ++i) {
        int64_t id;
        meta >> id;
        std::vector<float> vec(dim_);
        for (int d = 0; d < dim_; ++d) meta >> vec[d];
        id_to_vector_[id] = std::move(vec);
    }
    // 反向映射（带 REVMAP 哨兵，兼容旧版无反向映射的 meta 文件）
    std::string marker;
    if (meta >> marker && marker == "REVMAP") {
        id_to_discrete_.clear();
        size_t disc_cnt;
        meta >> disc_cnt;
        for (size_t i = 0; i < disc_cnt; ++i) {
            int64_t id;
            size_t fmap_size;
            meta >> id >> fmap_size;
            for (size_t j = 0; j < fmap_size; ++j) {
                std::string field, value;
                meta >> field >> value;
                id_to_discrete_[id][field] = value;
            }
        }
        id_to_continuous_bucket_.clear();
        size_t cont_cnt;
        meta >> cont_cnt;
        for (size_t i = 0; i < cont_cnt; ++i) {
            int64_t id;
            size_t fmap_size;
            meta >> id >> fmap_size;
            for (size_t j = 0; j < fmap_size; ++j) {
                std::string field;
                int bidx;
                meta >> field >> bidx;
                id_to_continuous_bucket_[id][field] = bidx;
            }
        }
    }
    // 字段配置（FIELDS 哨兵，重建子图结构；兼容无此段的旧 meta 文件）
    std::vector<std::string> loaded_disc_fields;
    std::vector<std::tuple<std::string, int64_t, int64_t, int>> loaded_cont_fields;
    if (meta >> marker && marker == "FIELDS") {
        size_t disc_field_cnt;
        meta >> disc_field_cnt;
        for (size_t i = 0; i < disc_field_cnt; ++i) {
            std::string field;
            meta >> field;
            loaded_disc_fields.push_back(field);
        }
        size_t cont_field_cnt;
        meta >> cont_field_cnt;
        for (size_t i = 0; i < cont_field_cnt; ++i) {
            std::string field;
            int64_t fmin, fmax;
            int bsize;
            meta >> field >> fmin >> fmax >> bsize;
            loaded_cont_fields.emplace_back(field, fmin, fmax, bsize);
        }
    }
    meta.close();

    // 重建子图结构（此前 Load 依赖外部 Register，现自洽）
    for (const auto& field : loaded_disc_fields) {
        if (!HasDiscreteSubgraph(field)) RegisterDiscreteField(field);
    }
    for (const auto& [field, fmin, fmax, bsize] : loaded_cont_fields) {
        if (!HasContinuousSubgraph(field)) RegisterContinuousField(field, fmin, fmax, bsize);
    }
    // 从反向映射反推离散子图的 value 集合，建空子图实例 + 恢复 discrete_counts_
    for (const auto& [id, fmap] : id_to_discrete_) {
        for (const auto& [field, value] : fmap) {
            if (discrete_subgraphs_.count(field)) {
                GetOrCreateDiscreteSubgraph(field, value);  // 建空子图（若不存在）
                discrete_counts_[field][value]++;           // 恢复 count
            }
        }
    }
    // 恢复连续桶 count（桶结构已由 RegisterContinuousField 建好）
    for (const auto& [id, fmap] : id_to_continuous_bucket_) {
        for (const auto& [field, bidx] : fmap) {
            auto it = continuous_fields_.find(field);
            if (it != continuous_fields_.end() && bidx >= 0 &&
                static_cast<size_t>(bidx) < it->second.buckets.size()) {
                it->second.buckets[bidx].count++;
            }
        }
    }

    // 子图数据从各自的 .index 文件加载（结构已重建，value_map/桶已就位）
    for (auto& [field, value_map] : discrete_subgraphs_) {
        for (auto& [value, index] : value_map) {
            if (index) {
                index->LoadIndex(path + "_disc_" + field + "_" + value + ".index");
            }
        }
    }
    for (auto& [field, config] : continuous_fields_) {
        for (size_t i = 0; i < config.buckets.size(); ++i) {
            if (config.buckets[i].index) {
                config.buckets[i].index->LoadIndex(
                    path + "_cont_" + field + "_" + std::to_string(i) + ".index");
            }
        }
    }
    global_logger->info("GardenIndex: loaded from {}", path);
}

}  // namespace vectordb
