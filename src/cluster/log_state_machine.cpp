#include "cluster/log_state_machine.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common/constants.h"
#include "common/vector_cfg.h"
#include "logger/logger.h"

namespace vectordb {

namespace {
// 从 JSON 请求中解析 collectionName（默认 "default"）
auto GetCollectionNameFromRequest(const rapidjson::Document& json_request) -> std::string {
    if (json_request.HasMember(REQUEST_COLLECTION_NAME) && json_request[REQUEST_COLLECTION_NAME].IsString()) {
        return json_request[REQUEST_COLLECTION_NAME].GetString();
    }
    return DEFAULT_COLLECTION_NAME;
}
} // anonymous namespace

void LogStateMachine::SetVectorDatabase(VectorDatabase *vector_database) {
  vector_database_ = vector_database;  // 设置 vector_database_ 指针
  last_committed_idx_ = vector_database->GetStartIndexId();
}

auto LogStateMachine::commit(const nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> {
  std::string content(reinterpret_cast<const char *>(data.data() + data.pos() + sizeof(int)),
                      data.size() - sizeof(int));
  global_logger->debug("Commit log_idx: {}, content: {}", log_idx, content);

  rapidjson::Document json_request;
  json_request.Parse(content.c_str());

  // Update last committed index number.
  last_committed_idx_ = log_idx;

  // 解析 collectionName
  std::string collection_name = GetCollectionNameFromRequest(json_request);

  std::string op_type;
  if (json_request.HasMember(REQUEST_OPERATION_TYPE) && json_request[REQUEST_OPERATION_TYPE].IsString()) {
    op_type = json_request[REQUEST_OPERATION_TYPE].GetString();
  }

  // 元数据变更：必须在每个副本上以相同顺序生效。
  // 复制状态机要求"初始状态相同 + 操作序列相同"，若建集合 / 注册 GARDEN 字段
  // 只在接收请求的节点上生效，各副本的状态机初始条件就不同，
  // 重放同一串 upsert 会得到不同的索引结构（GARDEN 连续子图在副本上根本不存在）。
  if (op_type == OPERATION_TYPE_CREATE_COLLECTION) {
    vector_database_->ApplyCreateCollection(json_request);
  } else if (op_type == OPERATION_TYPE_REGISTER_GARDEN_FIELD) {
    vector_database_->ApplyRegisterGardenField(json_request);
  } else if (op_type == OPERATION_TYPE_BATCH_UPSERT) {
    // 批量插入操作
    IndexFactory::IndexType index_type = vector_database_->GetIndexTypeFromRequest(json_request);
    if (json_request.HasMember(REQUEST_ITEMS) && json_request[REQUEST_ITEMS].IsArray()) {
      const auto& items = json_request[REQUEST_ITEMS].GetArray();
      std::vector<uint64_t> ids;
      std::vector<rapidjson::Document> datas;
      ids.reserve(items.Size());
      datas.reserve(items.Size());
      for (rapidjson::SizeType i = 0; i < items.Size(); ++i) {
        const auto& item = items[i];
        if (!item.IsObject() || !item.HasMember(REQUEST_ID) || !item.HasMember(REQUEST_VECTORS)) {
          continue;
        }
        uint64_t id = item[REQUEST_ID].GetUint64();
        rapidjson::Document doc;
        doc.CopyFrom(item, doc.GetAllocator());
        ids.push_back(id);
        datas.push_back(std::move(doc));
      }
      if (!ids.empty()) {
        vector_database_->BatchUpsert(collection_name, ids, datas, index_type);
      }
    }
  } else {
    // 单条插入操作
    uint64_t label = json_request[REQUEST_ID].GetUint64();
    IndexFactory::IndexType index_type = vector_database_->GetIndexTypeFromRequest(json_request);
    vector_database_->Upsert(collection_name, label, json_request, index_type);
  }

  // Return Raft log number as a return result.
  nuraft::ptr<nuraft::buffer> ret = nuraft::buffer::alloc(sizeof(log_idx));
  nuraft::buffer_serializer bs(ret);
  bs.put_u64(log_idx);
  return ret;
}

auto LogStateMachine::pre_commit(const nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> {
  std::string content(reinterpret_cast<const char *>(data.data() + data.pos() + sizeof(int)),
                      data.size() - sizeof(int));
  global_logger->debug("Pre Commit log_idx: {}, content: {}", log_idx, content);  // 添加打印日志
  return nullptr;
}

auto LogStateMachine::last_snapshot() -> nuraft::ptr<nuraft::snapshot> {
  std::lock_guard<std::mutex> l(snapshot_lock_);
  return last_snapshot_;
}

void LogStateMachine::create_snapshot(nuraft::snapshot &s,
                                      nuraft::async_result<bool>::handler_type &when_done) {
  // 把"Raft 快照"与"应用层索引快照"合并成同一个动作：
  // Raft 快照的语义是"到 s.get_last_log_idx() 为止的状态机状态已持久化"，
  // 而 Persistence::TakeSnapshot 正是把所有 collection 的索引落盘并记录该位点。
  // 两者此前互不知情：NuRaft 不知道应用层做过快照（于是永不 compact），
  // 应用层也不知道 Raft 的快照节奏（于是只能靠手动 RPC 触发）。
  const nuraft::ulong snap_idx = s.get_last_log_idx();

  bool ok = false;
  if (vector_database_ == nullptr) {
    global_logger->error("create_snapshot: vector_database_ is null, snapshot at idx {} skipped", snap_idx);
  } else {
    try {
      vector_database_->TakeSnapshot(static_cast<uint64_t>(snap_idx));
      ok = true;
    } catch (const std::exception &e) {
      global_logger->error("create_snapshot failed at log idx {}: {}", snap_idx, e.what());
    }
  }

  if (ok) {
    {
      std::lock_guard<std::mutex> l(snapshot_lock_);
      last_snapshot_ = nuraft::cs_new<nuraft::snapshot>(snap_idx, s.get_last_log_term(), s.get_last_config());
    }
    global_logger->debug("Created snapshot at log idx {}", snap_idx);
  }

  // 必须调用该回调。它是 NuRaft 快照流程的完成信号：
  // 不调 → on_snapshot_completed 不执行 → log_store_->compact_async 永不触发
  //      → snp_in_progress_ 一直是 true → 后续快照全部被跳过
  //      → Raft 日志只增不减，内存持续增长直至 OOM。
  nuraft::ptr<std::exception> exp(nullptr);
  when_done(ok, exp);
}

// ============================================================================
// 快照流式传输
// ============================================================================
//
// 载荷格式：[uint32 文件名长度][文件名][uint64 数据长度][数据]
//
// read_logical_snp_obj 在 leader 侧被调用（按 obj_id 逐个取），
// save_logical_snp_obj 在落后副本侧被调用（逐个落盘），
// 收完后 NuRaft 调 apply_snapshot 把快照加载进状态机。

namespace {
// 收集快照目录中属于"索引快照"的文件。
// raft_config_*/raft_state_* 是节点本地的 Raft 元数据（含 term 与集群配置），
// 绝不能通过快照传输覆盖到其它节点，否则会破坏选举安全性。
auto CollectSnapshotFiles() -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  const std::string snap_path = Cfg::Instance().SnapPath();
  std::error_code ec;
  if (snap_path.empty() || !std::filesystem::exists(snap_path, ec)) return files;
  for (const auto &entry : std::filesystem::directory_iterator(snap_path, ec)) {
    if (!entry.is_regular_file(ec)) continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind("raft_config_", 0) == 0 || name.rfind("raft_state_", 0) == 0) continue;
    files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  return files;
}
}  // namespace

auto LogStateMachine::read_logical_snp_obj(nuraft::snapshot &s, void *&user_snp_ctx, nuraft::ulong obj_id,
                                           nuraft::ptr<nuraft::buffer> &data_out, bool &is_last_obj) -> int {
  auto *files = static_cast<std::vector<std::filesystem::path> *>(user_snp_ctx);
  if (files == nullptr) {
    files = new std::vector<std::filesystem::path>(CollectSnapshotFiles());
    user_snp_ctx = files;
  }

  if (obj_id >= files->size()) {
    // 没有更多对象
    data_out = nuraft::buffer::alloc(0);
    is_last_obj = true;
    return 0;
  }

  const auto &path = (*files)[obj_id];
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    global_logger->error("read_logical_snp_obj: cannot open {}", path.string());
    return -1;
  }
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  const std::string name = path.filename().string();
  const uint32_t name_len = static_cast<uint32_t>(name.size());
  const uint64_t data_len = static_cast<uint64_t>(bytes.size());

  nuraft::ptr<nuraft::buffer> buf =
      nuraft::buffer::alloc(sizeof(name_len) + name.size() + sizeof(data_len) + bytes.size());
  buf->put(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
  if (!name.empty()) buf->put(name.data(), name.size());
  buf->put(reinterpret_cast<const char *>(&data_len), sizeof(data_len));
  if (!bytes.empty()) buf->put(bytes.data(), bytes.size());
  buf->pos(0);
  data_out = buf;

  is_last_obj = (obj_id + 1 >= files->size());
  global_logger->debug("read_logical_snp_obj: obj_id={} file={} bytes={} is_last={}",
                       obj_id, name, bytes.size(), is_last_obj);
  return 0;
}

void LogStateMachine::free_user_snp_ctx(void *&user_snp_ctx) {
  auto *files = static_cast<std::vector<std::filesystem::path> *>(user_snp_ctx);
  delete files;
  user_snp_ctx = nullptr;
}

void LogStateMachine::save_logical_snp_obj(nuraft::snapshot &s, nuraft::ulong &obj_id, nuraft::buffer &data,
                                           bool is_first_obj, bool is_last_obj) {
  const char *base = reinterpret_cast<const char *>(data.data_begin());
  const size_t total = data.size();
  size_t off = 0;

  auto fail = [&](const char *reason) {
    global_logger->error("save_logical_snp_obj (obj_id={}): {}", obj_id, reason);
  };

  if (total < sizeof(uint32_t)) {
    fail("payload too short");
    return;
  }
  uint32_t name_len = 0;
  std::memcpy(&name_len, base + off, sizeof(name_len));
  off += sizeof(name_len);
  if (off + name_len + sizeof(uint64_t) > total) {
    fail("truncated file name");
    return;
  }
  const std::string name(base + off, name_len);
  off += name_len;
  uint64_t data_len = 0;
  std::memcpy(&data_len, base + off, sizeof(data_len));
  off += sizeof(data_len);
  if (off + data_len > total) {
    fail("truncated file data");
    return;
  }

  const auto target = std::filesystem::path(Cfg::Instance().SnapPath()) / name;
  std::ofstream out(target, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    global_logger->error("save_logical_snp_obj: cannot write {}", target.string());
    return;
  }
  if (data_len > 0) {
    out.write(base + off, static_cast<std::streamsize>(data_len));
  }
  out.flush();
  global_logger->debug("save_logical_snp_obj: obj_id={} file={} bytes={} is_last={}",
                       obj_id, name, data_len, is_last_obj);
}

auto LogStateMachine::apply_snapshot(nuraft::snapshot &s) -> bool {
  if (vector_database_ == nullptr) {
    global_logger->error("apply_snapshot: vector_database_ is null");
    return false;
  }
  // 所有 object 已落盘，这里把它们加载进内存状态机
  vector_database_->LoadSnapshot();
  {
    std::lock_guard<std::mutex> l(snapshot_lock_);
    last_snapshot_ =
        nuraft::cs_new<nuraft::snapshot>(s.get_last_log_idx(), s.get_last_log_term(), s.get_last_config());
  }
  global_logger->info("Applied snapshot at log idx {}", s.get_last_log_idx());
  return true;
}

}  // namespace vectordb