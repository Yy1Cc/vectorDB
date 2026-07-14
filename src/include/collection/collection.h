#pragma once

#include <string>
#include "index/index_factory.h"
#include "index/fulltext_index.h"
#include "database/ttl_manager.h"

namespace vectordb {

// Collection 元数据：描述一个 Collection 的配置
struct CollectionMeta {
    std::string name;
    int dimension = 0;
    int num_data = 0;
    IndexFactory::MetricType metric = IndexFactory::MetricType::L2;
    std::string created_at;
};

// Collection：一个独立的向量集合，拥有自己的索引实例集、全文索引和 TTL 管理器
// 每个 Collection 可以有不同的维度、距离度量和索引类型配置
class Collection {
public:
    CollectionMeta meta;
    IndexFactory index_factory;
    FullTextIndex fulltext_index;
    TTLManager ttl_manager;

    // 便捷方法：委托给 index_factory
    auto GetIndex(IndexFactory::IndexType type) const -> void* {
        return index_factory.GetIndex(type);
    }

    // 初始化该 Collection 的所有索引实例
    void InitAllIndices() {
        int dim = meta.dimension;
        int num_data = meta.num_data;

        index_factory.Init(IndexFactory::IndexType::FLAT, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::HNSW, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::FILTER, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::SQ8, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::SQ4, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::IP_FLAT, dim, num_data, IndexFactory::MetricType::IP);
        index_factory.Init(IndexFactory::IndexType::IP_SQ8, dim, num_data, IndexFactory::MetricType::IP);
        index_factory.Init(IndexFactory::IndexType::LAYERED_FLAT, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::LAYERED_SQ8, dim, num_data);
        index_factory.Init(IndexFactory::IndexType::GARDEN_HNSW, dim, num_data);
    }
};

}  // namespace vectordb
