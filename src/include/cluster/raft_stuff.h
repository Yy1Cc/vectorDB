#pragma once

#include <libnuraft/asio_service.hxx>
#include <string>
#include <vector>
#include "cluster/in_memory_state_mgr.h"
#include "log_state_machine.h"
#include "logger/logger.h"  // 包含 logger.h 以使用日志记录器
namespace vectordb {

// Raft 组内节点的信息快照。
// is_learner 供 Master 判定谁需要自动转正：learner 不计入法定人数、
// 也不发起选举（NuRaft handle_timeout.cxx:280），若不转正会永久丧失补位能力。
struct RaftNodeInfo {
    int node_id = 0;
    std::string endpoint;
    std::string role;  // "leader" / "follower"
    bool is_learner = false;
    nuraft::ulong last_log_idx = 0;
    nuraft::ulong last_succ_resp_us = 0;
};

static const nuraft::raft_params::return_method_type CALL_TYPE
    = nuraft::raft_params::blocking;
//  = nuraft::raft_params::async_handler;
class RaftStuff {
 public:
  RaftStuff(int node_id, std::string &endpoint, int port, VectorDatabase *vector_database);

  void Init();
  // as_learner=true：新节点不计入法定人数，可后台追赶而不阻塞线上写入。
  // 默认 false 是为了兼容既有调用；生产环境加节点应显式用 learner 再加入后转正。
  auto AddSrv(int srv_id, const std::string &srv_endpoint, bool as_learner = false) -> bool;
  // 将 learner 提升为正式成员。必须在它追平 leader 日志后调用，
  // 否则法定人数立刻提高，新写入会被未追上的节点卡住直至超时。
  auto PromoteLearner(int srv_id) -> bool;
  // 查询对端是否已成为正式成员（learner 标记为 false）
  auto IsVotingMember(int srv_id) const -> bool;
  // 查询对端已追到的日志位置，用于判断 learner 是否追平
  auto GetPeerLastLogIdx(int srv_id) const -> nuraft::ulong;
  // 本节点已提交的日志位置
  auto GetCommittedLogIdx() const -> nuraft::ulong;
  // 将节点从 Raft 组中摘除（缩容）。必须由 leader 发起，且拒绝摘除自身。
  auto RemoveSrv(int srv_id) -> bool;
  void EnableElectionTimeout(int lower_bound, int upper_bound);  // 定义 enableElectionTimeout 方法
  auto IsLeader() const -> bool;                                 // 添加 isLeader 方法声明
  auto GetAllNodesInfo() const -> std::vector<RaftNodeInfo>;
  auto GetCurrentNodesInfo() const -> std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong>;
  auto GetNodeStatus(int node_id) const -> std::string;  // 添加 getNodeStatus 方法声明
  // 返回 true 表示日志已达成共识并提交；false 表示非 leader / 被拒绝 / 超时。
  // 调用方必须据此设置响应码，否则写入失败会被静默吞掉（历史上 upsert 无条件返回成功）。
  auto AppendEntries(const std::string &entry) -> bool;
  auto GetSrvConfig(int srv_id) -> nuraft::ptr<nuraft::srv_config>;
  auto HandleResult(nuraft::cmd_result< nuraft::ptr<nuraft::buffer> >& result) -> bool;
 private:
 
 private:
  int node_id_;
  std::string endpoint_;
  nuraft::ptr<nuraft::state_mgr> smgr_;
  nuraft::ptr<nuraft::state_machine> sm_;
  int port_;
  nuraft::raft_launcher launcher_;
  nuraft::ptr<nuraft::raft_server> raft_instance_;
  VectorDatabase *vector_database_;  // 添加一个 VectorDatabase 指针成员变量
};

}  // namespace vectordb