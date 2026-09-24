#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <cstdint> // 包含 <cstdint> 以使用 uint64_t 类型
#include <rapidjson/document.h> // 包含 rapidjson/document.h 以使用 JSON 对象
#include <snappy/snappy.h>
#include "index/index_factory.h"
#include "common/vector_cfg.h"
namespace vectordb {

class Persistence {
public:
    Persistence();
    ~Persistence();

    void Init(const std::string& local_path); // 添加 init 方法声明
    auto IncreaseId() -> uint64_t;
    auto GetId() const -> uint64_t;
    void WriteWalLog(const std::string& operation_type, const rapidjson::Document& json_data, const std::string& version); // 添加 version 参数
    void WriteWalRawLog(uint64_t log_id, const std::string& operation_type, const std::string& raw_data, const std::string& version); // 添加 writeWALRawLog 函数声明
    void ReadNextWalLog(std::string* operation_type, rapidjson::Document* json_data); // 更改返回类型为 void 并添加指针参数
    void TakeSnapshot(); 
    // 按 Raft 日志位点做快照：snapshot_log_id 会写入 MaxLogID，
    // 重启时 ReadNextWalLog 据此跳过已快照的 WAL 条目。
    // 供 LogStateMachine::create_snapshot 使用，让 Raft 快照与索引快照成为同一个动作。
    void TakeSnapshot(uint64_t snapshot_log_id);
    void LoadSnapshot(); // 添加 loadSnapshot 方法声明
    void SaveLastSnapshotId(const std::string& folder_path); // 添加 saveLastSnapshotID 方法声明
    void LoadLastSnapshotId(const std::string& folder_path); // 添加 loadLastSnapshotID 方法声明

    // Raft 日志压缩后调用：丢弃 log_id <= last_log_id 的 WAL 条目。
    // 前提是这些条目的状态已被快照覆盖（MaxLogID 已持久化），否则会造成数据丢失。
    void TruncateWalBefore(uint64_t last_log_id);


private:
    // 从一条压缩后的 WAL 记录里解出 log_id；解析失败返回 false（调用方保守保留该条目）。
    static auto ParseWalLogId(const std::string& compressed_line, uint64_t* out_id) -> bool;

    uint64_t increase_id_;
    uint64_t last_snapshot_id_; // 添加 lastSnapshotID_ 成员变量
    std::fstream wal_log_file_; // 将 wal_log_file_ 类型更改为 std::fstream
    std::string wal_path_;         // WAL 文件路径，截断时需要关闭后重开
    mutable std::mutex wal_mutex_; // 保护 wal_log_file_（写入来自 Raft 线程，截断来自 compact 线程）
};

}  // namespace vectordb