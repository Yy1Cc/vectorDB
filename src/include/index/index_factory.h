#pragma once

#include "faiss_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexScalarQuantizer.h"
#include "common/vector_utils.h"
#include <map>

namespace vectordb {
class IndexFactory {
public:
    enum class IndexType {
        FLAT,
        HNSW,
        FILTER, // 添加 FILTER 枚举值
        SQ8,    // 8bit 标量量化，4x 压缩
        SQ4,    // 4bit 标量量化， 8x 压缩
        IP_FLAT, // 内积(余弦)索引，ip2cos 预处理
        IP_SQ8,  // 内积(余弦)+SQ8 量化
        LAYERED_FLAT, // 分层存储（streaming part + FLAT）
        LAYERED_SQ8,  // 分层存储（streaming part + SQ8）
        GARDEN_HNSW,  // GARDEN 标量过滤引擎（多子图 + Pruner/Grafter/Selector）
        UNKNOWN = -1
    };

    enum class MetricType {
        L2,
        IP
    };

    IndexFactory() = default;
    ~IndexFactory() = default;

    void Init(IndexType type, int dim,  int num_data, MetricType metric = MetricType::L2);
    auto GetIndex(IndexType type) const -> void*;
     void SaveIndex(const std::string& folder_path); // 添加 ScalarStorage 参数
    void LoadIndex(const std::string& folder_path); // 添加 loadIndex 方法声明

    // 该 Collection 的向量条数。以 FLAT 索引为准（所有写入路径都会维护它）；
    // 若 FLAT 未初始化则退化为 HNSW，都没有则为 0。用于负载上报与再平衡决策。
    auto GetTotalCount() const -> int64_t;



private:

    std::map<IndexType, void*> index_map_; 

};

}  // namespace vectordb