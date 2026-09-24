#include "index/index_factory.h"
#include <fstream>
#include "index/hnswlib_index.h"
#include "index/filter_index.h"
#include "index/layered_index.h"
#include "index/garden_index.h"
#include "logger/logger.h"
namespace vectordb {

void IndexFactory::Init(IndexType type, int dim,  int num_data,MetricType metric) {
    faiss::MetricType faiss_metric = (metric == MetricType::L2) ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

    switch (type) {
        case IndexType::FLAT:
            index_map_[type] = new vectordb::FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss_metric)));
            break;
        case IndexType::HNSW:
            index_map_[type] = new vectordb::HNSWLibIndex(dim, num_data, metric, 16, 200);
            break;
        case IndexType::FILTER: // 初始化 FilterIndex 对象
            index_map_[type] = new FilterIndex();
            break;
        case IndexType::SQ8: // 8bit 标量量化，4x 压缩
            index_map_[type] = new vectordb::FaissIndex(new faiss::IndexIDMap(new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss_metric)));
            break;
        case IndexType::SQ4: // 4bit 标量量化，8x 压缩
            index_map_[type] = new vectordb::FaissIndex(new faiss::IndexIDMap(new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_4bit, faiss_metric)));
            break;
        case IndexType::IP_FLAT: // 内积(余弦)索引，ip2cos 预处理
            index_map_[type] = new vectordb::FaissIndex(
                new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);
            break;
        case IndexType::IP_SQ8: // 内积(余弦)+SQ8 量化
            index_map_[type] = new vectordb::FaissIndex(
                new faiss::IndexIDMap(new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_INNER_PRODUCT)), true);
            break;
        case IndexType::LAYERED_FLAT: { // 分层存储：streaming part + FLAT
            auto* base = new vectordb::FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss_metric)));
            index_map_[type] = new vectordb::LayeredIndex(base, dim, metric == MetricType::IP);
            break;
        }
        case IndexType::LAYERED_SQ8: { // 分层存储：streaming part + SQ8
            auto* base = new vectordb::FaissIndex(new faiss::IndexIDMap(new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss_metric)));
            index_map_[type] = new vectordb::LayeredIndex(base, dim, metric == MetricType::IP);
            break;
        }
        case IndexType::GARDEN_HNSW: // GARDEN 标量过滤引擎
            index_map_[type] = new vectordb::GardenIndex(dim, num_data, metric);
            break;
        default:
            break;
    }
}

auto IndexFactory::GetIndex(IndexType type) const -> void* { 
    auto it = index_map_.find(type);
    if (it != index_map_.end()) {
        return it->second;
    }
    return nullptr;
}

void IndexFactory::MarkUsed(IndexType type) {
    // 仅首次标记时打日志，避免高频刷屏
    if (used_types_.insert(type).second) {
        global_logger->debug("IndexFactory: index type {} marked as used, will be persisted in snapshots",
                             static_cast<int>(type));
    }
}

auto IndexFactory::GetUsedTypes() const -> std::set<IndexType> {
    return used_types_;
}

auto IndexFactory::UsedTypesFilePath(const std::string& folder_path) -> std::string {
    return folder_path + "used_types.bin";
}

void IndexFactory::SaveUsedTypes(const std::string& folder_path) const {
    const std::string file_path = UsedTypesFilePath(folder_path);
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        // 静默失败会让重启后 used_types_ 丢失 → 快照漏写 → 数据丢失，必须显式报错
        global_logger->error("IndexFactory: cannot write used_types sidecar: {}", file_path);
        return;
    }
    const uint32_t count = static_cast<uint32_t>(used_types_.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (IndexType type : used_types_) {
        const int32_t v = static_cast<int32_t>(type);
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    out.flush();
}

auto IndexFactory::LoadUsedTypes(const std::string& folder_path) -> bool {
    std::ifstream in(UsedTypesFilePath(folder_path), std::ios::binary);
    if (!in.is_open()) {
        // 旧格式快照没有 sidecar，由调用方兜底为"全部类型"
        return false;
    }
    uint32_t count = 0;
    if (!in.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        global_logger->error("IndexFactory: truncated used_types sidecar header: {}", UsedTypesFilePath(folder_path));
        return false;
    }
    used_types_.clear();
    for (uint32_t i = 0; i < count; ++i) {
        int32_t v = 0;
        if (!in.read(reinterpret_cast<char*>(&v), sizeof(v))) {
            global_logger->error("IndexFactory: truncated used_types sidecar entry: {}", UsedTypesFilePath(folder_path));
            return false;
        }
        used_types_.insert(static_cast<IndexType>(v));
    }
    return true;
}

auto IndexFactory::GetTypeCount(IndexType type) const -> int64_t {
    auto it = index_map_.find(type);
    if (it == index_map_.end() || it->second == nullptr) {
        return -1;
    }
    switch (type) {
        case IndexType::FLAT:
        case IndexType::SQ8:
        case IndexType::SQ4:
        case IndexType::IP_FLAT:
        case IndexType::IP_SQ8:
            return static_cast<FaissIndex*>(it->second)->GetTotalCount();
        case IndexType::HNSW:
            return static_cast<HNSWLibIndex*>(it->second)->GetTotalCount();
        case IndexType::LAYERED_FLAT:
        case IndexType::LAYERED_SQ8:
            return static_cast<LayeredIndex*>(it->second)->GetTotalCount();
        case IndexType::GARDEN_HNSW:
            return static_cast<int64_t>(static_cast<GardenIndex*>(it->second)->GetTotalCount());
        case IndexType::FILTER: // FilterIndex 没有向量计数语义
        default:
            return -1;
    }
}

void IndexFactory::SaveIndex(const std::string& folder_path) {
    // 只落盘实际被写入过的索引类型。每个 collection 预建 10 种索引，
    // 但业务写入通常只命中其中一种；全量落盘会让单次快照写 >1GB，
    // 而快照在 commit 线程上同步执行，会阻塞提交十余秒。
    // FILTER 无条件写入：GARDEN 搜索与标量过滤依赖它构造 bitmap。
    used_types_.insert(IndexType::FILTER);

    for (const auto& index_entry : index_map_) {
        IndexType index_type = index_entry.first;
        void* index = index_entry.second;
        if (index == nullptr) continue;
        if (used_types_.find(index_type) == used_types_.end()) continue;

        // 为每个索引类型生成一个文件名
        std::string file_path = folder_path + std::to_string(static_cast<int>(index_type)) + ".index";

        // 根据索引类型调用相应的 saveIndex 函数
        if (index_type == IndexType::FLAT) {
            static_cast<FaissIndex*>(index)->SaveIndex(file_path);
        } else if (index_type == IndexType::HNSW) {
            static_cast<HNSWLibIndex*>(index)->SaveIndex(file_path);
        } else if (index_type == IndexType::FILTER) { // 保存 FilterIndex 类型的索引
            static_cast<FilterIndex*>(index)->SaveIndex(file_path);
        } else if (index_type == IndexType::SQ8 || index_type == IndexType::SQ4 ||
                   index_type == IndexType::IP_FLAT || index_type == IndexType::IP_SQ8) { // 保存量化/IP索引（复用 FaissIndex）
            static_cast<FaissIndex*>(index)->SaveIndex(file_path);
        } else if (index_type == IndexType::LAYERED_FLAT || index_type == IndexType::LAYERED_SQ8) { // 保存分层索引
            static_cast<LayeredIndex*>(index)->SaveIndex(file_path);
        } else if (index_type == IndexType::GARDEN_HNSW) { // 保存 GARDEN 索引
            static_cast<GardenIndex*>(index)->SaveIndex(file_path);
        }
    }

    SaveUsedTypes(folder_path);
}

void IndexFactory::LoadIndex(const std::string& folder_path) {
    // 先恢复 used_types_。sidecar 缺失说明是旧格式快照：
    // 不知道哪些类型有数据，必须先全部加载（否则可能漏加载），
    // 加载完再把没有数据的类型从 used_types_ 剔除，
    // 否则一次旧快照加载会让此后每次快照都退回全量落盘。
    const bool has_sidecar = LoadUsedTypes(folder_path);
    if (!has_sidecar) {
        for (const auto& entry : index_map_) {
            used_types_.insert(entry.first);
        }
    }

    for (const auto& index_entry : index_map_) {
        IndexType index_type = index_entry.first;
        void* index = index_entry.second;
        if (index == nullptr) continue;
        if (used_types_.find(index_type) == used_types_.end()) continue;

        // 为每个索引类型生成一个文件名
        std::string file_path = folder_path + std::to_string(static_cast<int>(index_type)) + ".index";

        // 根据索引类型调用相应的 loadIndex 函数
        if (index_type == IndexType::FLAT) {
            static_cast<FaissIndex*>(index)->LoadIndex(file_path);
        } else if (index_type == IndexType::HNSW) {
            static_cast<HNSWLibIndex*>(index)->LoadIndex(file_path);
        } else if (index_type == IndexType::FILTER) { // 加载 FilterIndex 类型的索引
            static_cast<FilterIndex*>(index)->LoadIndex(file_path);
        } else if (index_type == IndexType::SQ8 || index_type == IndexType::SQ4 ||
                   index_type == IndexType::IP_FLAT || index_type == IndexType::IP_SQ8) { // 加载量化/IP索引（复用 FaissIndex）
            static_cast<FaissIndex*>(index)->LoadIndex(file_path);
        } else if (index_type == IndexType::LAYERED_FLAT || index_type == IndexType::LAYERED_SQ8) { // 加载分层索引
            static_cast<LayeredIndex*>(index)->LoadIndex(file_path);
        } else if (index_type == IndexType::GARDEN_HNSW) { // 加载 GARDEN 索引
            static_cast<GardenIndex*>(index)->LoadIndex(file_path);
        }
    }

    if (!has_sidecar) {
        // 剪枝：FILTER 保留，其余剔除无数据的类型
        for (auto it = used_types_.begin(); it != used_types_.end();) {
            if (*it == IndexType::FILTER || GetTypeCount(*it) > 0) {
                ++it;
            } else {
                it = used_types_.erase(it);
            }
        }
        // 立即固化剪枝结果，避免下次快照再走一遍全量
        SaveUsedTypes(folder_path);
        global_logger->info("IndexFactory: legacy snapshot loaded, used_types pruned to {} entries",
                            used_types_.size());
    }
}

auto IndexFactory::GetTotalCount() const -> int64_t {
    // 按实际被写入过的类型取计数，而非硬编码 FLAT → HNSW。
    // InitAllIndices 会建全部 10 种索引，FLAT 对象必然存在且非 null；
    // 业务若实际使用 HNSW / GARDEN，旧实现会命中 FLAT 的空索引并恒返回 0，
    // 导致负载上报（vectorCount）与再平衡决策失效。
    for (IndexType type : used_types_) {
        if (type == IndexType::FILTER) continue;
        const int64_t n = GetTypeCount(type);
        if (n > 0) return n;
    }

    // used_types_ 为空（尚未写入 / 旧快照兜底前）时保留旧行为
    auto it = index_map_.find(IndexType::FLAT);
    if (it != index_map_.end() && it->second != nullptr) {
        return static_cast<FaissIndex*>(it->second)->GetTotalCount();
    }
    it = index_map_.find(IndexType::HNSW);
    if (it != index_map_.end() && it->second != nullptr) {
        return static_cast<HNSWLibIndex*>(it->second)->GetTotalCount();
    }
    return 0;
}

}  // namespace vectordb