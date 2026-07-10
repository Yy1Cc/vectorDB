#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "collection/collection.h"
#include "common/vector_utils.h"

namespace vectordb {

// CollectionManager：管理所有 Collection 的生命周期（单例）
// 提供运行时动态创建/删除/查询 Collection 的能力
// Collection 元数据持久化到 JSON 文件，服务重启后自动恢复
class CollectionManager : public Singleton<CollectionManager> {
    friend class Singleton<CollectionManager>;

public:
    // 创建 Collection（运行时动态创建）
    // name: Collection 名称（唯一标识）
    // dim: 向量维度
    // num_data: 预估向量数量（用于 HNSW 索引初始化）
    // metric: 距离度量（L2 或 IP）
    bool CreateCollection(const std::string& name, int dim, int num_data,
                          IndexFactory::MetricType metric = IndexFactory::MetricType::L2);

    // 删除 Collection（释放索引资源）
    bool DropCollection(const std::string& name);

    // 获取 Collection（不存在返回 nullptr）
    Collection* GetCollection(const std::string& name);

    // 列出所有 Collection 名称
    std::vector<std::string> ListCollections() const;

    // 检查 Collection 是否存在
    bool HasCollection(const std::string& name) const;

    // 获取 Collection 元数据
    bool GetCollectionMeta(const std::string& name, CollectionMeta& meta) const;

    // 持久化：将所有 Collection 元数据保存到 JSON 文件
    void SaveToDisk(const std::string& path);

    // 恢复：从 JSON 文件加载所有 Collection 元数据并重建索引实例
    void LoadFromDisk(const std::string& path);

private:
    CollectionManager() = default;
    ~CollectionManager() = default;

    std::map<std::string, std::unique_ptr<Collection>> collections_;
    mutable std::mutex mutex_;
};

}  // namespace vectordb
