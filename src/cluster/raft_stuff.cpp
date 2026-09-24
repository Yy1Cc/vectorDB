#include "cluster/raft_stuff.h"
#include "cluster/raft_logger_wrapper.h"
#include "logger/logger.h"

namespace vectordb {

RaftStuff::RaftStuff(int node_id, std::string &endpoint, int port, VectorDatabase *vector_database)
    : node_id_(node_id),
      endpoint_(endpoint),
      port_(port),
      vector_database_(vector_database) {  // 初始化 vector_database_ 指针
  Init();
}

void RaftStuff::Init() {
  smgr_ = nuraft::cs_new<InmemStateMgr>(node_id_, endpoint_, vector_database_);
  sm_ = nuraft::cs_new<LogStateMachine>();

  // 将 state_machine 对象强制转换为 log_state_machine 对象
  nuraft::ptr<LogStateMachine> log_sm = std::dynamic_pointer_cast<LogStateMachine>(sm_);

  log_sm->SetVectorDatabase(
      vector_database_);  // 将 vector_database_ 参数传递给 log_state_machine 的 setVectorDatabase 函数

  nuraft::asio_service::options asio_opt;
  // 该线程池承载全部 Raft RPC：心跳、日志复制、快照传输、投票。
  // 原值 1 意味着任一 RPC 处理变慢（尤其是快照传输整文件读入内存）都会
  // 让心跳停摆，follower 超时后触发选举把 leader 推翻。
  asio_opt.thread_pool_size_ = 4;

  // nuraft::raft_params params;
  // params.election_timeout_lower_bound_ = 100000000;  // 设置为一个非常大的值
  // params.election_timeout_upper_bound_ = 200000000;  // 设置为一个非常大的值

  // Raft parameters.
  nuraft::raft_params params;
#if defined(WIN32) || defined(_WIN32)
  // heartbeat: 1 sec, election timeout: 2 - 4 sec.
  params.heart_beat_interval_ = 1000;
  params.election_timeout_lower_bound_ = 2000;
  params.election_timeout_upper_bound_ = 4000;
#else
  // heartbeat: 100 ms, election timeout: 200 - 400 ms.
  params.heart_beat_interval_ = 100;
  params.election_timeout_lower_bound_ = 200;
  params.election_timeout_upper_bound_ = 400;
#endif
  // 每次快照都要遍历所有 collection 全量写索引（Persistence::TakeSnapshot），
  // 代价远高于一次日志追加。原值 5 意味着每 5 条写入就全量落盘一遍索引，
  // 会让写入性能直接崩溃。改为 10 万量级：既保证日志能被压缩
  // （避免 Raft 日志只增不减导致 OOM），又把快照开销摊薄到可忽略。
  params.reserved_log_items_ = 1000;
  params.snapshot_distance_ = 100000;
  // 需能覆盖一次快照停摆。快照在 commit 线程上同步执行（非独立线程），
  // 期间 sm_commit_index_ 不推进，客户端的 blocking append_entries 会一直等。
  // 10 秒兜不住 GB 级索引落盘，配合 TakeSnapshot 减量后仍需放宽。
  params.client_req_timeout_ = 60000;
  // According to this method, `append_log` function
  // should be handled differently.
  params.return_method_ = CALL_TYPE;

  // 快照对象的读取（read_logical_snp_obj）改由后台线程异步执行。
  // 默认 false 时它由 Raft worker 线程同步读，整文件读入内存（GB 级）期间
  // 该线程无法处理任何 RPC —— 包括心跳 —— 会被 follower 误判为 leader 失联。
  // 该参数在 NuRaft 中标注 Experimental，但默认组合的风险明确且严重得多。
  params.use_bg_thread_for_snapshot_io_ = true;

  // Logger.
  std::string log_file_name = "./srv" + std::to_string(node_id_) + ".log";
  nuraft::ptr<LoggerWrapper> log_wrap = nuraft::cs_new<LoggerWrapper>(log_file_name);

  raft_instance_ = launcher_.init(sm_, smgr_, log_wrap, port_, asio_opt, params);

  if (!raft_instance_) {
    global_logger->error("Failed to initialize launcher (see the message in the log file)");
    log_wrap.reset();
    exit(-1);
  }

  // Wait until Raft server is ready (upto 5 seconds).
  const size_t max_try = 100;
  global_logger->info("init Raft instance ");
  for (size_t ii = 0; ii < max_try; ++ii) {
    if (raft_instance_->is_initialized()) {
      global_logger->info("done");
      global_logger->debug("RaftStuff initialized with node_id: {}, endpoint: {}, port: {}", node_id_, endpoint_,
                           port_);  // 添加打印日志
      return;
    }
    global_logger->info(".");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  global_logger->error("RaftStuff Init FAILED");
  log_wrap.reset();
  exit(-1);
}

auto RaftStuff::AddSrv(int srv_id, const std::string &srv_endpoint, bool as_learner) -> bool {
  bool success = false;
  nuraft::ptr<nuraft::srv_config> peer_srv_conf = nuraft::cs_new<nuraft::srv_config>(srv_id, srv_endpoint);
  if (as_learner) {
    // NuRaft 的 `use_new_joiner_type_` 默认关闭，新节点会立即计入法定人数。
    // 显式标记为 learner 才能让它后台追赶而不影响线上写入。
    peer_srv_conf->set_learner(true);
    global_logger->info("Adding server srv_id={} endpoint={} as LEARNER (excluded from quorum)",
                        srv_id, srv_endpoint);
  } else {
    global_logger->warn("Adding server srv_id={} as full member: quorum will grow immediately, "
                        "writes may block until it catches up", srv_id);
  }
  auto ret = raft_instance_->add_srv(*peer_srv_conf);

  if (!ret->get_accepted()) {
    global_logger->error("raft_stuff  AddSrv  failed");
    return false;
  }

  // Wait until it appears in server list.
  const size_t max_try = 40;
  for (size_t jj = 0; jj < max_try; ++jj) {
    global_logger->info("Wait for add follower.");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    auto conf = GetSrvConfig(srv_id);
    if (conf) {
      success = true;
      global_logger->info(" Add follower done.");
      break;
    }
  }
  return success;
}

auto RaftStuff::PromoteLearner(int srv_id) -> bool {
  if (!raft_instance_) {
    global_logger->error("PromoteLearner: Raft instance is not available");
    return false;
  }
  if (!raft_instance_->is_leader()) {
    global_logger->error("PromoteLearner: node {} is not the leader", node_id_);
    return false;
  }
  auto conf = GetSrvConfig(srv_id);
  if (!conf) {
    global_logger->error("PromoteLearner: srv {} not in the raft group", srv_id);
    return false;
  }
  if (!conf->is_learner()) {
    global_logger->info("PromoteLearner: srv {} is already a voting member", srv_id);
    return true;
  }

  auto ret = raft_instance_->flip_learner_flag(srv_id, false);
  if (!ret || ret->get_result_code() != nuraft::cmd_result_code::OK) {
    global_logger->error("PromoteLearner: flip_learner_flag failed for srv {}", srv_id);
    return false;
  }

  // 等待配置生效
  const size_t max_try = 40;
  for (size_t jj = 0; jj < max_try; ++jj) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    auto c = GetSrvConfig(srv_id);
    if (c && !c->is_learner()) {
      global_logger->info("Promoted srv {} to voting member", srv_id);
      return true;
    }
  }
  global_logger->error("PromoteLearner: srv {} promotion not confirmed", srv_id);
  return false;
}

auto RaftStuff::IsVotingMember(int srv_id) const -> bool {
  if (!raft_instance_) return false;
  auto conf = raft_instance_->get_srv_config(srv_id);
  return conf && !conf->is_learner();
}

auto RaftStuff::GetPeerLastLogIdx(int srv_id) const -> nuraft::ulong {
  if (!raft_instance_) return 0;
  nuraft::raft_server::peer_info info = raft_instance_->get_peer_info(srv_id);
  return info.last_log_idx_;
}

auto RaftStuff::GetCommittedLogIdx() const -> nuraft::ulong {
  if (!raft_instance_) return 0;
  return raft_instance_->get_committed_log_idx();
}

auto RaftStuff::RemoveSrv(int srv_id) -> bool {
  if (!raft_instance_) {
    global_logger->error("Cannot remove srv: Raft instance is not available");
    return false;
  }
  if (!raft_instance_->is_leader()) {
    global_logger->error("Cannot remove srv {}: current node is not the leader", srv_id);
    return false;
  }
  if (srv_id == node_id_) {
    // leader 把自己摘掉会导致集群失去写入能力, 属于误用, 直接拒绝
    global_logger->error("Refuse to remove self (node {}) from raft group", srv_id);
    return false;
  }
  if (!GetSrvConfig(srv_id)) {
    global_logger->warn("Srv {} is not in the raft group, treat as removed", srv_id);
    return true;
  }

  global_logger->info("Removing srv {} from raft group", srv_id);
  auto ret = raft_instance_->remove_srv(srv_id);
  if (!ret || !ret->get_accepted()) {
    global_logger->error("raft_stuff RemoveSrv failed for srv {}", srv_id);
    return false;
  }

  // Wait until it disappears from server list.
  bool success = false;
  const size_t max_try = 40;
  for (size_t jj = 0; jj < max_try; ++jj) {
    global_logger->info("Wait for remove follower.");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    auto conf = GetSrvConfig(srv_id);
    if (!conf) {
      success = true;
      global_logger->info(" Remove follower done.");
      break;
    }
  }
  if (!success) {
    global_logger->warn("Remove follower {} not confirmed within {} ms", srv_id, max_try * 250);
  }
  return success;
}

auto RaftStuff::GetSrvConfig(int srv_id) -> nuraft::ptr<nuraft::srv_config> {
  global_logger->debug("get server config with srv_id: {}", srv_id);  // 添加打印日志
  return raft_instance_->get_srv_config(srv_id);
}

auto RaftStuff::AppendEntries(const std::string &entry) -> bool {
  if (!raft_instance_) {
    global_logger->error("Cannot append entries: Raft instance is not available");
    return false;
  }
  if (!raft_instance_->is_leader()) {
    // 非 leader 无法复制日志。调用方必须据此返回错误，
    // 否则请求被静默丢弃、客户端却拿到成功响应。
    global_logger->error("Cannot append entries: current node (id={}) is not the leader", node_id_);
    return false;
  }

  // 计算所需的内存大小
  size_t total_size = sizeof(int) + entry.size();

  // 创建一个 Raft 日志条目
  nuraft::ptr<nuraft::buffer> log_entry_buffer = nuraft::buffer::alloc(total_size);
  nuraft::buffer_serializer bs_log(log_entry_buffer);

  bs_log.put_str(entry);

  // 将日志条目追加到 Raft 实例中
  auto ret = raft_instance_->append_entries({log_entry_buffer});

  if (!ret->get_accepted()) {
    // Log append rejected, usually because this node is not a leader.
    global_logger->error("Failed append log: result_code={}", static_cast<int>(ret->get_result_code()));
    return false;
  }
  // Log append accepted, but that doesn't mean the log is committed.
  // Commit result can be obtained below.

  if (CALL_TYPE == nuraft::raft_params::blocking) {
    // Blocking mode:
    //   `append_entries` returns after getting a consensus,
    //   so that `ret` already has the result from state machine.
    return HandleResult(*ret);

  } else if (CALL_TYPE == nuraft::raft_params::async_handler) {
    // Async mode:
    //   `append_entries` returns immediately, commit result is not available yet.
    //   此处只能判定"已被受理"，最终结果由回调里的 HandleResult 处理。
    global_logger->debug("Append log accepted (async mode), commit result pending");
    ret->when_ready(std::bind(&RaftStuff::HandleResult, this, std::placeholders::_1));
    return true;
  }
  assert(0);
  return false;
}

void RaftStuff::EnableElectionTimeout(int lower_bound, int upper_bound) {
  if (raft_instance_) {
    nuraft::raft_params params = raft_instance_->get_current_params();
    params.election_timeout_lower_bound_ = lower_bound;
    params.election_timeout_upper_bound_ = upper_bound;
    raft_instance_->update_params(params);
  }
}

auto RaftStuff::IsLeader() const -> bool {
  if (!raft_instance_) {
    return false;
  }
  return raft_instance_->is_leader();  // 调用 raft_instance_ 的 is_leader() 方法
}

auto RaftStuff::GetAllNodesInfo() const
    -> std::vector<std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong>> {
  std::vector<std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong>> nodes_info;

  if (!raft_instance_) {
    global_logger->warn("raft_instance empty");
    return nodes_info;
  }

  // 获取配置信息

  // get_srv_config_all
  // auto config = raft_instance_->get_config();
  // if (!config) {
  //   return nodes_info;
  // }

  // 获取服务器列表
  // auto servers = config->get_servers();

  std::vector<nuraft::ptr<nuraft::srv_config>> configs;
  raft_instance_->get_srv_config_all(configs);

  int leader_id = raft_instance_->get_leader();
  for (auto &entry : configs) {
    nuraft::ptr<nuraft::srv_config> &srv = entry;
    // 获取节点状态
    std::string node_state;
    if (srv->get_id() == leader_id) {
      node_state = "leader";
    } else {
      node_state = "follower";
    }

    // 使用正确的类型
    nuraft::raft_server::peer_info node_info = raft_instance_->get_peer_info(srv->get_id());
    nuraft::ulong last_log_idx = node_info.last_log_idx_;
    nuraft::ulong last_succ_resp_us = node_info.last_succ_resp_us_;

    nodes_info.emplace_back(
        std::make_tuple(srv->get_id(), srv->get_endpoint(), node_state, last_log_idx, last_succ_resp_us));
  }
  return nodes_info;
}

auto RaftStuff::GetCurrentNodesInfo() const -> std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong> {
  std::tuple<int, std::string, std::string, nuraft::ulong, nuraft::ulong> nodes_info;

  if (!raft_instance_) {
    return nodes_info;
  }

  // 获取配置信息
  auto config = raft_instance_->get_config();
  if (!config) {
    return nodes_info;
  }

  // 获取服务器列表
  auto servers = config->get_servers();

  for (const auto &srv : servers) {
    if (srv && srv->get_id() == node_id_) {
      // 获取节点状态
      std::string node_state;
      if (srv->get_id() == raft_instance_->get_leader()) {
        node_state = "leader";
      } else {
        node_state = "follower";
      }

      // 使用正确的类型
      nuraft::raft_server::peer_info node_info = raft_instance_->get_peer_info(srv->get_id());
      nuraft::ulong last_log_idx = node_info.last_log_idx_;
      nuraft::ulong last_succ_resp_us = node_info.last_succ_resp_us_;
      nodes_info = std::make_tuple(srv->get_id(), srv->get_endpoint(), node_state, last_log_idx, last_succ_resp_us);
      break;
    }
  }

  return nodes_info;
}

auto RaftStuff::HandleResult(nuraft::cmd_result<nuraft::ptr<nuraft::buffer>> &result) -> bool {
  if (result.get_result_code() != nuraft::cmd_result_code::OK) {
    // Something went wrong.
    // This means committing this log failed,
    // but the log itself is still in the log store.
    // 必须向上返回 false，否则调用方会让客户端误以为写入成功。
    global_logger->error("Raft commit failed: result_code={}", static_cast<int>(result.get_result_code()));
    return false;
  }
  global_logger->debug("Raft commit succeeded");
  return true;
}

}  // namespace vectordb