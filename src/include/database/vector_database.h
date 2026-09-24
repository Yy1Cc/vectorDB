#pragma once

#include "database/scalar_storage.h"
#include "index/index_factory.h"
#include "index/fulltext_index.h"
#include "database/ttl_manager.h"
#include <string>
#include <vector>
#include <rapidjson/document.h>
#include "database/persistence.h"
namespace vectordb {

class VectorDatabase {
public:
    // 构造函数
    explicit VectorDatabase(const std::string& db_path,const std::string& wal_path);

    // 插入或更新向量（collection_name 指定目标 Collection）
    void Upsert(const std::string& collection_name, uint64_t id, const rapidjson::Document& data, IndexFactory::IndexType index_type);

    // 批量插入或更新向量
    void BatchUpsert(const std::string& collection_name, const std::vector<uint64_t>& ids, const std::vector<rapidjson::Document>& datas, IndexFactory::IndexType index_type);
    auto Query(const std::string& collection_name, uint64_t id) -> rapidjson::Document;
    auto Search(const std::string& collection_name, const rapidjson::Document& json_request) -> std::pair<std::vector<int64_t>, std::vector<float>>;
    void ReloadDatabase(); // 添加 reloadDatabase 方法声明
    void WriteWalLog(const std::string& operation_type, const rapidjson::Document& json_data); // 添加 writeWALLog 方法声明
    void WriteWalLogWithId(uint64_t log_id, const std::string& data);
    auto GetIndexTypeFromRequest(const rapidjson::Document& json_request) -> vectordb::IndexFactory::IndexType;
    void TakeSnapshot();
    // 按 Raft 日志位点做快照（供 LogStateMachine::create_snapshot 调用）
    void TakeSnapshot(uint64_t snapshot_log_id);
    // Raft 日志重放侧的元数据变更入口（供 LogStateMachine::commit 调用）。
    // 这两个操作此前只在接收请求的节点上本地生效，导致各副本状态机初始条件不一致。
    // 实现必须幂等：日志可能被重复重放（重启、补日志）。
    auto ApplyCreateCollection(const rapidjson::Document& json_request) -> bool;
    auto ApplyRegisterGardenField(const rapidjson::Document& json_request) -> bool;

    // Raft 日志压缩后清理对应的 WAL 条目（供 InmemLogStore::compact 调用）
    void TruncateWalBefore(uint64_t last_log_id);
    auto GetStartIndexId() const -> int64_t; // 添加 getStartIndexID 函数声明

    // TTL 清理：删除所有过期向量（遍历所有 Collection）
    void CleanExpiredVectors();

private:
    ScalarStorage scalar_storage_;
    Persistence persistence_;
};

}  // namespace vectordb