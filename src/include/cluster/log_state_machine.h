#pragma once

#include <libnuraft/nuraft.hxx>
#include <mutex>
#include "database/vector_database.h"
namespace vectordb {
class LogStateMachine : public nuraft::state_machine {
 public:
  LogStateMachine() : last_committed_idx_(0), vector_database_(nullptr) {}
  ~LogStateMachine() override = default;
  void SetVectorDatabase(VectorDatabase *vector_database);  // 重命名为 setVectorDatabase
  auto commit(nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> override;

  auto pre_commit(nuraft::ulong log_idx, nuraft::buffer &data) -> nuraft::ptr<nuraft::buffer> override;
  void commit_config(const nuraft::ulong log_idx, nuraft::ptr<nuraft::cluster_config> &new_conf) override {
    // Nothing to do with configuration change. Just update committed index.
    last_committed_idx_ = log_idx;
  }
  void rollback(const nuraft::ulong log_idx, nuraft::buffer &data) override {}
  // 快照流式传输三件套 + 上下文释放。
  //
  // 这三个接口是落后副本（尤其是 learner）在自身需要的日志已被 compact 掉时
  // 唯一能追上来的通道。此前全是空实现：read_logical_snp_obj 直接 return 0
  // 等于告诉 NuRaft"没有快照可发"，于是新节点只能靠重放全部日志追赶。
  //
  // 传输单位：SnapPath 目录下的每个快照文件作为一个 object，
  // 载荷格式为 [uint32 文件名长度][文件名][uint64 数据长度][数据]。
  auto read_logical_snp_obj(nuraft::snapshot &s, void *&user_snp_ctx, nuraft::ulong obj_id,
                            nuraft::ptr<nuraft::buffer> &data_out, bool &is_last_obj) -> int override;
  void save_logical_snp_obj(nuraft::snapshot &s, nuraft::ulong &obj_id, nuraft::buffer &data, bool is_first_obj,
                            bool is_last_obj) override;
  auto apply_snapshot(nuraft::snapshot &s) -> bool override;
  void free_user_snp_ctx(void *&user_snp_ctx) override;
  auto last_snapshot() -> nuraft::ptr<nuraft::snapshot> override;
  auto last_commit_index() -> nuraft::ulong override { return last_committed_idx_; }

  void create_snapshot(nuraft::snapshot &s, nuraft::async_result<bool>::handler_type &when_done) override;

 private:
  // Last committed Raft log number.
  std::atomic<uint64_t> last_committed_idx_;
  VectorDatabase *vector_database_;  // 添加一个 VectorDatabase 指针成员变量

  // 最近一次成功创建的快照。NuRaft 用它决定 compact 到哪个位置。
  // 返回 nullptr 会让 NuRaft 认为"从来没有快照"，从而永不触发日志压缩。
  mutable std::mutex snapshot_lock_;
  nuraft::ptr<nuraft::snapshot> last_snapshot_;
};

}  // namespace vectordb