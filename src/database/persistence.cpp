#include "database/persistence.h"
#include <chrono>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "common/vector_utils.h"
#include "collection/collection_manager.h"
#include "logger/logger.h"
namespace vectordb {

Persistence::Persistence() : increase_id_(10), last_snapshot_id_(0) {}

Persistence::~Persistence() {
  if (wal_log_file_.is_open()) {
    wal_log_file_.close();
  }
}

void Persistence::Init(const std::string &local_path) {
  // 确保 WAL 与快照目录都存在。目录缺失会让快照写入静默失败
  // （ofstream 打开失败只打日志），是难以排查的隐性故障。
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(local_path).parent_path(), ec);
  const std::string& snap_path = Cfg::Instance().SnapPath();
  if (!snap_path.empty()) {
    std::filesystem::create_directories(snap_path, ec);
  }

  if (!std::filesystem::exists(local_path)) {
    // 文件不存在，先创建文件
    std::ofstream temp_file(local_path);
    temp_file.close();
  }

  wal_path_ = local_path;
  wal_log_file_.open(local_path, std::ios::in | std::ios::out |
                                     std::ios::app);  // 以 std::ios::in | std::ios::out | std::ios::app 模式打开文件
  if (!wal_log_file_.is_open()) {
    global_logger->error("An error occurred while writing the WAL log entry. Reason: {}",
                         std::strerror(errno));  // 使用日志打印错误消息和原因
    throw std::runtime_error("Failed to open WAL log file at path: " + local_path);
  }

  LoadLastSnapshotId(Cfg::Instance().SnapPath());
}

auto Persistence::IncreaseId() -> uint64_t {
  increase_id_++;
  return increase_id_;
}

auto Persistence::GetId() const -> uint64_t { return increase_id_; }

void Persistence::WriteWalLog(const std::string &operation_type, const rapidjson::Document &json_data,
                              const std::string &version) {  // 添加 version 参数
  uint64_t log_id = IncreaseId();

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json_data.Accept(writer);

  std::string json_data_str = buffer.GetString();
  // 拼接日志条目
  std::ostringstream oss;
  oss << log_id << "|" << version << "|" << operation_type << "|" << json_data_str;

  // 压缩日志条目
  std::string compressed_data;
  snappy::Compress(oss.str().c_str(), oss.str().size(), &compressed_data);

  // 写入压缩后的日志条目到文件
  wal_log_file_ << compressed_data << std::endl;

  if (wal_log_file_.fail()) {  // 检查是否发生错误
    global_logger->error("An error occurred while writing the WAL log entry. Reason: {}",
                         std::strerror(errno));  // 使用日志打印错误消息和原因
  } else {
    global_logger->debug("Wrote WAL log entry: log_id={}, version={}, operation_type={}, json_data_str={}", log_id,
                         version, operation_type, buffer.GetString());  // 打印日志
    wal_log_file_.flush();                                              // 强制持久化
  }
}

void Persistence::WriteWalRawLog(uint64_t log_id, const std::string &operation_type, const std::string &raw_data,
                                 const std::string &version) {
  // 拼接日志条目
  std::ostringstream oss;
  oss << log_id << "|" << version << "|" << operation_type << "|" << raw_data;

  // 压缩日志条目
  std::string compressed_data;
  snappy::Compress(oss.str().c_str(), oss.str().size(), &compressed_data);

  // 写入压缩后的日志条目到文件（与 TruncateWalBefore 互斥，后者会关闭并重建文件）
  std::lock_guard<std::mutex> l(wal_mutex_);
  wal_log_file_ << compressed_data << std::endl;

  if (wal_log_file_.fail()) {  // 检查是否发生错误
    global_logger->error("An error occurred while writing the WAL raw log entry. Reason: {}",
                         std::strerror(errno));  // 使用日志打印错误消息和原因
  } else {
    global_logger->debug("Wrote WAL raw log entry: log_id={}, version={}, operation_type={}, raw_data={}", log_id,
                         version, operation_type, raw_data);  // 打印日志
    wal_log_file_.flush();                                    // 强制持久化
  }
}

void Persistence::ReadNextWalLog(std::string *operation_type, rapidjson::Document *json_data) {
  global_logger->debug("Reading next WAL log entry");

  std::string compressed_line;
  while (std::getline(wal_log_file_, compressed_line)) {
    std::string decompressed_data;
    if (!snappy::Uncompress(compressed_line.c_str(), compressed_line.size(), &decompressed_data)) {
      global_logger->error("Failed to decompress WAL log entry");
      return;
    }

    // 解析解压后的日志条目
    std::istringstream iss(decompressed_data);

    std::string log_id_str;
    std::string version;
    std::string json_data_str;

    std::getline(iss, log_id_str, '|');
    std::getline(iss, version, '|');
    std::getline(iss, *operation_type, '|');  // 使用指针参数返回 operation_type
    std::getline(iss, json_data_str, '|');

    uint64_t log_id = std::stoull(log_id_str);  // 将 log_id_str 转换为 uint64_t 类型
    if (log_id > increase_id_) {                // 如果 log_id 大于当前 increase_id_
      increase_id_ = log_id;                    // 更新 increase_id_
    }

    if (log_id > last_snapshot_id_) {
      json_data->Parse(json_data_str.c_str());  // 使用指针参数返回 json_data
      global_logger->debug("Read WAL log entry: log_id={}, operation_type={}, json_data_str={}", log_id_str,
                           *operation_type, json_data_str);
      return;
    }
    // TODO(zhouzj): 增加last_snapshot_id_ 前WAL LOG的清除
    global_logger->debug("Skip Read WAL log entry: log_id={}, operation_type={}, json_data_str={}", log_id_str,
                         *operation_type, json_data_str);
  }
  operation_type->clear();
  wal_log_file_.clear();
  global_logger->debug("No more WAL log entries to read");
}

void Persistence::TakeSnapshot() {
  TakeSnapshot(increase_id_);
}

void Persistence::TakeSnapshot(uint64_t snapshot_log_id) {
  // 快照在 NuRaft 的 commit 线程上同步执行，期间 sm_commit_index_ 不推进、
  // 客户端写入被挂起。耗时日志是观测"停摆时长"的直接手段，必须有。
  const auto t0 = std::chrono::steady_clock::now();
  global_logger->info("TakeSnapshot begin at log id {}", snapshot_log_id);

  last_snapshot_id_ = snapshot_log_id;
  std::string snapshot_folder_path = Cfg::Instance().SnapPath();
  // 遍历所有 Collection，保存各自的索引
  auto names = CollectionManager::Instance().ListCollections();
  for (const auto& name : names) {
    auto* coll = CollectionManager::Instance().GetCollection(name);
    if (coll != nullptr) {
      const auto c0 = std::chrono::steady_clock::now();
      std::string coll_path = snapshot_folder_path + name + "_";
      coll->index_factory.SaveIndex(coll_path);
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - c0)
                          .count();
      global_logger->info("TakeSnapshot: collection '{}' saved in {} ms", name, ms);
    }
  }
  SaveLastSnapshotId(snapshot_folder_path);

  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
  global_logger->info("TakeSnapshot end at log id {}, {} collections, {} ms total", snapshot_log_id,
                      names.size(), total_ms);
}

void Persistence::LoadSnapshot() {
  global_logger->debug("Loading snapshot");
  std::string snapshot_folder_path = Cfg::Instance().SnapPath();
  // 遍历所有 Collection，加载各自的索引
  auto names = CollectionManager::Instance().ListCollections();
  for (const auto& name : names) {
    auto* coll = CollectionManager::Instance().GetCollection(name);
    if (coll != nullptr) {
      std::string coll_path = snapshot_folder_path + name + "_";
      coll->index_factory.LoadIndex(coll_path);
    }
  }
}

// 快照位点文件名。保存端与加载端必须使用同一个名字 —— 此前两端分别是
// "MaxLogID" 和 ".MaxLogID"，且都把字符串字面量 "file_path" 当成了文件名，
// 导致 last_snapshot_id_ 永远读不回来（恒为 0），重启时 WAL 被全量重放，
// 快照相当于白做。
namespace {
constexpr const char* kSnapshotIdFileName = "MaxLogID";
}  // namespace

void Persistence::SaveLastSnapshotId(const std::string &folder_path) {
  const std::string file_path = folder_path + kSnapshotIdFileName;
  std::ofstream file(file_path);
  if (!file.is_open()) {
    global_logger->error("Failed to open snapshot MaxLogID file for writing: {}", file_path);
    return;
  }
  file << last_snapshot_id_;
  file.close();
  global_logger->debug("save snapshot Max log ID {} to {}", last_snapshot_id_, file_path);
}

void Persistence::LoadLastSnapshotId(const std::string &folder_path) {
  const std::string file_path = folder_path + kSnapshotIdFileName;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    // 首次启动或尚未做过快照：从 0 开始，重放全部 WAL。
    global_logger->info("Snapshot MaxLogID file not found ({}), replaying full WAL", file_path);
    last_snapshot_id_ = 0;
    return;
  }
  file >> last_snapshot_id_;
  file.close();
  global_logger->debug("Loading snapshot Max log ID {} from {}", last_snapshot_id_, file_path);
}

auto Persistence::ParseWalLogId(const std::string& compressed_line, uint64_t* out_id) -> bool {
  std::string raw;
  if (!snappy::Uncompress(compressed_line.data(), compressed_line.size(), &raw)) {
    return false;
  }
  const auto pos = raw.find('|');
  if (pos == std::string::npos || pos == 0) {
    return false;
  }
  try {
    *out_id = std::stoull(raw.substr(0, pos));
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

void Persistence::TruncateWalBefore(uint64_t last_log_id) {
  std::lock_guard<std::mutex> l(wal_mutex_);
  if (wal_path_.empty() || !wal_log_file_.is_open()) {
    return;
  }
  wal_log_file_.flush();
  wal_log_file_.close();

  // 读出全部条目，只保留 log_id > last_log_id 的部分。
  // 无法解析的条目一律保守保留，避免因格式问题误删数据。
  std::vector<std::string> keep;
  {
    std::ifstream in(wal_path_, std::ios::binary);
    if (!in.is_open()) {
      global_logger->error("TruncateWalBefore: cannot reopen WAL for reading: {}", wal_path_);
      return;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      uint64_t id = 0;
      if (!ParseWalLogId(line, &id) || id > last_log_id) {
        keep.push_back(line);
      }
    }
  }

  {
    std::ofstream out(wal_path_, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      global_logger->error("TruncateWalBefore: cannot reopen WAL for writing: {}", wal_path_);
      return;
    }
    for (const auto& entry : keep) {
      out << entry << "\n";
    }
    out.flush();
  }

  // 恢复 append 模式，否则后续写入会全部失败（且是静默失败）
  wal_log_file_.open(wal_path_, std::ios::in | std::ios::out | std::ios::app);
  if (!wal_log_file_.is_open()) {
    global_logger->error("TruncateWalBefore: failed to reopen WAL in append mode: {}", wal_path_);
    return;
  }

  global_logger->info("Truncated WAL before log id {}: kept {} entries", last_log_id, keep.size());
}

}  // namespace vectordb