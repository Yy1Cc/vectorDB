#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "index_factory.h"
namespace vectordb {
class HNSWLibIndex {
public:
    // 构造函数
    HNSWLibIndex(int dim, int num_data, IndexFactory::MetricType metric, int M = 16, int ef_construction = 200); // 将MetricType参数修改为第三个参数

    // 插入向量
    void InsertVectors(const std::vector<float>& data, int64_t label);

    // 批量插入向量
    void BatchInsertVectors(const std::vector<float>& data, int n, const std::vector<int64_t>& labels);

    // 查询向量
    auto SearchVectors(const std::vector<float>& query, int k, const roaring_bitmap_t* bitmap = nullptr,int ef_search = 50) -> std::pair<std::vector<int64_t>, std::vector<float>>;

    void RemoveVectors(const std::vector<int64_t>& ids);

    void SaveIndex(const std::string& file_path); // 添加 saveIndex 方法声明
    void LoadIndex(const std::string& file_path); // 添加 loadIndex 方法声明

    // 内存序列化：用于 GARDEN 的单文件容器格式与 Raft 快照流式传输。
    // hnswlib 只暴露基于文件路径的 saveIndex/loadIndex，这里借道临时文件
    // 转成内存 buffer，把实现细节屏蔽在索引层内部。
    // Deserialize 失败时抛出 std::runtime_error。
    auto Serialize() -> std::vector<char>;
    void Deserialize(const char* data, size_t size);

    // 当前索引中的向量条数。用于负载上报与再平衡决策。
    auto GetTotalCount() const -> int64_t;

        // 定义 RoaringBitmapIDFilter 类
    class RoaringBitmapIDFilter : public hnswlib::BaseFilterFunctor {
    public:
        explicit RoaringBitmapIDFilter(const roaring_bitmap_t* bitmap) : bitmap_(bitmap) {}

        auto operator()(hnswlib::labeltype label) -> bool override {
            return roaring_bitmap_contains(bitmap_, static_cast<uint32_t>(label));
        }

    private:
        const roaring_bitmap_t* bitmap_;
    };
    
private:
    // int dim_;
    hnswlib::SpaceInterface<float>* space_;
    hnswlib::HierarchicalNSW<float>* index_;
    size_t max_elements_; // 添加 max_elements 成员变量
};
}  // namespace vectordb

