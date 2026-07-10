#include "index/layered_index.h"
#include <algorithm>
#include "logger/logger.h"

namespace vectordb {

// ========== StreamingPart ==========

StreamingPart::StreamingPart(int dim, bool is_ip, int flush_threshold)
    : dim_(dim), is_ip_(is_ip), flush_threshold_(flush_threshold) {}

void StreamingPart::Insert(const std::vector<float>& vec, int64_t id) {
    data_.insert(data_.end(), vec.begin(), vec.end());
    ids_.push_back(id);
}

void StreamingPart::BatchInsert(const std::vector<float>& data, int n, const std::vector<int64_t>& labels) {
    data_.insert(data_.end(), data.begin(), data.end());
    ids_.insert(ids_.end(), labels.begin(), labels.end());
}

auto StreamingPart::Search(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
    if (ids_.empty()) {
        return {std::vector<int64_t>(k, -1), std::vector<float>(k, 0.0f)};
    }

    int n = static_cast<int>(ids_.size());
    std::vector<std::pair<float, int64_t>> dist_id;
    dist_id.reserve(n);

    for (int i = 0; i < n; ++i) {
        // bitmap 过滤
        if (bitmap != nullptr && !roaring_bitmap_contains(bitmap, static_cast<uint32_t>(ids_[i]))) {
            continue;
        }

        float dist = 0.0f;
        const float* vec = &data_[static_cast<size_t>(i) * dim_];
        if (is_ip_) {
            for (int j = 0; j < dim_; ++j) dist += query[j] * vec[j];
        } else {
            for (int j = 0; j < dim_; ++j) {
                float diff = query[j] - vec[j];
                dist += diff * diff;
            }
        }
        dist_id.push_back({dist, ids_[i]});
    }

    if (dist_id.empty()) {
        return {std::vector<int64_t>(k, -1), std::vector<float>(k, 0.0f)};
    }

    int result_k = std::min(k, static_cast<int>(dist_id.size()));
    if (is_ip_) {
        std::partial_sort(dist_id.begin(), dist_id.begin() + result_k, dist_id.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::partial_sort(dist_id.begin(), dist_id.begin() + result_k, dist_id.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    std::vector<int64_t> result_ids(k, -1);
    std::vector<float> result_dists(k, 0.0f);
    for (int i = 0; i < result_k; ++i) {
        result_ids[i] = dist_id[i].second;
        result_dists[i] = dist_id[i].first;
    }
    return {result_ids, result_dists};
}

void StreamingPart::GetData(std::vector<float>& data, std::vector<int64_t>& ids) const {
    data = data_;
    ids = ids_;
}

void StreamingPart::Clear() {
    data_.clear();
    ids_.clear();
}

void StreamingPart::Remove(int64_t id) {
    auto it = std::find(ids_.begin(), ids_.end(), id);
    if (it != ids_.end()) {
        int idx = static_cast<int>(std::distance(ids_.begin(), it));
        ids_.erase(it);
        data_.erase(data_.begin() + idx * dim_, data_.begin() + (idx + 1) * dim_);
    }
}

// ========== LayeredIndex ==========

LayeredIndex::LayeredIndex(FaissIndex* base_index, int dim, bool is_ip, int flush_threshold)
    : base_index_(base_index), dim_(dim), is_ip_(is_ip),
      streaming_part_(std::make_unique<StreamingPart>(dim, is_ip, flush_threshold)) {}

void LayeredIndex::Insert(const std::vector<float>& vec, int64_t id) {
    std::vector<float> normalized = vec;
    if (is_ip_) { // IP 模式：归一化后存入 streaming part，保证和 base index distance 一致
        InnerProductSpace::Normalize(normalized);
    }
    streaming_part_->Insert(normalized, id);
    if (streaming_part_->NeedFlush()) {
        Flush();
    }
}

void LayeredIndex::BatchInsert(const std::vector<float>& data, int n, const std::vector<int64_t>& labels) {
    std::vector<float> normalized = data;
    if (is_ip_) {
        InnerProductSpace::NormalizeBatch(normalized, n, dim_);
    }
    streaming_part_->BatchInsert(normalized, n, labels);
    if (streaming_part_->NeedFlush()) {
        Flush();
    }
}

auto LayeredIndex::Search(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
    // 搜索 streaming part（IP 模式下用归一化后的 query）
    std::vector<float> normalized_query = query;
    if (is_ip_) {
        InnerProductSpace::Normalize(normalized_query);
    }
    auto [stream_ids, stream_dists] = streaming_part_->Search(normalized_query, k, bitmap);

    // 搜索 base index
    auto [base_ids, base_dists] = base_index_->SearchVectors(query, k, bitmap);

    // 合并两个结果
    std::vector<std::pair<float, int64_t>> merged;
    for (size_t i = 0; i < stream_ids.size(); ++i) {
        if (stream_ids[i] != -1) {
            merged.push_back({stream_dists[i], stream_ids[i]});
        }
    }
    for (size_t i = 0; i < base_ids.size(); ++i) {
        if (base_ids[i] != -1) {
            merged.push_back({base_dists[i], base_ids[i]});
        }
    }

    // 去重（同一个 ID 可能同时出现在 streaming part 和 base index 中）
    std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    auto last = std::unique(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
        return a.second == b.second;
    });
    merged.erase(last, merged.end());

    // 按距离排序
    if (is_ip_) {
        std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // 取前 k 个
    std::vector<int64_t> result_ids(k, -1);
    std::vector<float> result_dists(k, 0.0f);
    int result_k = std::min(k, static_cast<int>(merged.size()));
    for (int i = 0; i < result_k; ++i) {
        result_ids[i] = merged[i].second;
        result_dists[i] = merged[i].first;
    }
    return {result_ids, result_dists};
}

void LayeredIndex::Flush() {
    if (streaming_part_->Size() == 0) return;

    std::vector<float> data;
    std::vector<int64_t> ids;
    streaming_part_->GetData(data, ids);

    base_index_->BatchInsertVectors(data, static_cast<int>(ids.size()), ids);
    streaming_part_->Clear();

    global_logger->info("LayeredIndex flushed {} vectors to base index", ids.size());
}

LayeredIndex::~LayeredIndex() {
    delete base_index_;
}

void LayeredIndex::RemoveVectors(const std::vector<int64_t>& ids) {
    for (int64_t id : ids) {
        streaming_part_->Remove(id);
    }
    base_index_->RemoveVectors(ids);
}

void LayeredIndex::SaveIndex(const std::string& file_path) {
    Flush(); // 先把 streaming part 刷入 base index
    base_index_->SaveIndex(file_path);
}

void LayeredIndex::LoadIndex(const std::string& file_path) {
    base_index_->LoadIndex(file_path);
    streaming_part_->Clear();
}

}  // namespace vectordb
