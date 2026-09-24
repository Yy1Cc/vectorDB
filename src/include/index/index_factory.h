#pragma once

#include "faiss_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexScalarQuantizer.h"
#include "common/vector_utils.h"
#include <map>
#include <set>

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

    // 标记某个索引类型已被写入过。快照只落盘被标记过的类型。
    // 必须持久化：若 used_types_ 只存内存，重启后退化为空集合会导致快照漏写，
    // 而对应的 Raft 日志已被 compact 删除 —— 数据将永久丢失。
    void MarkUsed(IndexType type);
    auto GetUsedTypes() const -> std::set<IndexType>;

    // 该 Collection 的向量条数。按实际被写入过的索引类型（used_types_）取值，
    // 而非硬编码 FLAT：InitAllIndices 会预建全部 10 种索引，FLAT 对象必然存在，
    // 业务若实际使用 HNSW / GARDEN，按 FLAT 取值会恒返回 0。
    // 用于负载上报（vectorCount）与再平衡决策。
    auto GetTotalCount() const -> int64_t;



private:

    // sidecar 文件路径：{folder_path}used_types.bin，二进制 [uint32 count][int32 type]*
    static auto UsedTypesFilePath(const std::string& folder_path) -> std::string;
    void SaveUsedTypes(const std::string& folder_path) const;
    // 返回 false 表示 sidecar 不存在（旧格式快照），调用方应兜底为"全部类型"
    auto LoadUsedTypes(const std::string& folder_path) -> bool;
    // 单一索引类型的向量计数；类型未知或无计数语义（如 FILTER）返回 -1
    auto GetTypeCount(IndexType type) const -> int64_t;

    std::map<IndexType, void*> index_map_;
    std::set<IndexType> used_types_;

};

}  // namespace vectordb