#pragma once

#include "index/faiss_index.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "roaring/roaring.h"

namespace vectordb {

// StreamingPart: 内存中的暴力检索层
// 新写入的数据先进入此层，达到阈值后 flush 到正式索引
class StreamingPart {
public:
    StreamingPart(int dim, bool is_ip, int flush_threshold = 1000);

    void Insert(const std::vector<float>& vec, int64_t id);
    void BatchInsert(const std::vector<float>& data, int n, const std::vector<int64_t>& labels);

    // 暴力检索 top-k
    auto Search(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap = nullptr)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    bool NeedFlush() const { return static_cast<int>(ids_.size()) >= flush_threshold_; }
    void GetData(std::vector<float>& data, std::vector<int64_t>& ids) const;
    void Clear();
    void Remove(int64_t id);
    size_t Size() const { return ids_.size(); }

private:
    int dim_;
    bool is_ip_; // true: IP metric (distance 越大越好), false: L2 metric (distance 越小越好)
    int flush_threshold_;
    std::vector<float> data_;  // 连续存储 n*dim
    std::vector<int64_t> ids_;
};

// LayeredIndex: 分层索引 = StreamingPart + FaissIndex
// 写入先入 streaming part，搜索时合并两部分结果
class LayeredIndex {
public:
    LayeredIndex(FaissIndex* base_index, int dim, bool is_ip = false, int flush_threshold = 1000);
    ~LayeredIndex();

    void Insert(const std::vector<float>& vec, int64_t id);
    void BatchInsert(const std::vector<float>& data, int n, const std::vector<int64_t>& labels);

    auto Search(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap = nullptr)
        -> std::pair<std::vector<int64_t>, std::vector<float>>;

    // 和 FaissIndex 一致的方法名（用于 vector_database.cpp switch 复用）
    void InsertVectors(const std::vector<float>& data, int64_t label) { Insert(data, label); }
    void BatchInsertVectors(const std::vector<float>& data, int n, const std::vector<int64_t>& labels) { BatchInsert(data, n, labels); }
    auto SearchVectors(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap = nullptr)
        -> std::pair<std::vector<int64_t>, std::vector<float>> { return Search(query, k, bitmap); }
    void RemoveVectors(const std::vector<int64_t>& ids);
    void Train(int /*n*/, const std::vector<float>& /*data*/) {} // no-op，LayeredIndex 不需要训练

    void SaveIndex(const std::string& file_path);
    void LoadIndex(const std::string& file_path);

    void Flush(); // 把 streaming part 刷入 base index
    size_t StreamingSize() const { return streaming_part_->Size(); }

private:
    FaissIndex* base_index_;
    int dim_;
    bool is_ip_;
    std::unique_ptr<StreamingPart> streaming_part_;
};

}  // namespace vectordb
