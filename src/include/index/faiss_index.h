#pragma once

#include <faiss/impl/IDSelector.h>
#include "faiss/Index.h"
#include <faiss/utils/utils.h>
#include <cstdint>
#include <vector>
#include "roaring/roaring.h"
#include "index/inner_product_space.h"
namespace vectordb {

    // 定义 RoaringBitmapIDSelector 结构体
struct RoaringBitmapIDSelector : faiss::IDSelector {
    explicit RoaringBitmapIDSelector(const roaring_bitmap_t* bitmap) : bitmap_(bitmap) {}

    auto is_member(int64_t id) const -> bool final;

    ~RoaringBitmapIDSelector() override = default;

    const roaring_bitmap_t* bitmap_;
};
class FaissIndex {
public:
    explicit FaissIndex(faiss::Index* index, bool normalize = false);
    void InsertVectors(const std::vector<float>& data, int64_t label);
    void BatchInsertVectors(const std::vector<float>& data, int n, const std::vector<int64_t>& labels);
    auto SearchVectors(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap = nullptr) -> std::pair<std::vector<int64_t>, std::vector<float>>;
    void RemoveVectors(const std::vector<int64_t>& ids);
    void Train(int n, const std::vector<float>& data); // 训练量化索引（SQ8/SQ4 需要）
    void SaveIndex(const std::string& file_path); // 添加 saveIndex 方法声明
    void LoadIndex(const std::string& file_path); // 将返回类型更改为 faiss::Index*

    // 索引中当前的向量条数（faiss 的 ntotal）。用于负载上报与再平衡决策。
    auto GetTotalCount() const -> int64_t;
private:
    faiss::Index* index_;
    bool normalize_; // ip2cos: 为 true 时插入/查询前做 L2 归一化
};
}  // namespace vectordb
