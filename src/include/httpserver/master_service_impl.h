#pragma once
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include "brpc/stream.h"
#include "cluster/balancer.h"
#include "cluster/etcd_http_client.h"
#include "cluster/raft_stuff.h"
#include "database/vector_database.h"
#include "gmock/gmock.h"
#include "http.pb.h"
#include "httpserver/base_service_impl.h"
#include "index/faiss_index.h"
#include "index/index_factory.h"
namespace vectordb {

enum class ServerRole { Master, Backup };

struct ServerInfo {
  std::string url_;
  ServerRole role_;
  auto ToJson() const -> rapidjson::Document;
  static auto FromJson(const rapidjson::Document &value) -> ServerInfo;
};

struct Partition {
  uint64_t partition_id_;
  uint64_t node_id_;
};

struct PartitionConfig {
  std::string partition_key_;
  int number_of_partitions_;
  std::list<Partition> partitions_;  // 使用 std::list 存储分区信息
};

class MasterServiceImpl : public nvm::MasterService, public BaseServiceImpl {
 public:
  explicit MasterServiceImpl(const std::string &etcdEndpoints) : etcd_client_(etcdEndpoints){};

  ~MasterServiceImpl() override = default;

  void GetNodeInfo(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                   ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void AddNode(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
               ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void RemoveNode(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                  ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void GetInstance(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                   ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void GetPartitionConfig(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                          ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void UpdatePartitionConfig(::google::protobuf::RpcController *controller, const ::nvm::HttpRequest * /*request*/,
                             ::nvm::HttpResponse * /*response*/, ::google::protobuf::Closure *done) override;

  void UpdateNodeStates();

  // 再平衡执行器：评估各实例的分区分布，必要时把一个分区从最忙节点迁给最闲节点。
  // 由 master_server 的 10 秒后台循环驱动，内部自带单飞与限流。
  void RunBalancer();

 private:
  auto DoGetPartitionConfig(uint64_t instanceId) -> PartitionConfig;
  void DoUpdatePartitionConfig(uint64_t instanceId, const std::string &partitionKey, int numberOfPartitions,
                               const std::list<Partition> &partitions);

  // 从 leader 采集各节点的 Raft 日志进度，写入节点表，用于判断新节点是否已完成追平；
  // 同时把已注册但尚未进入 Raft 组的存活节点拉入组内
  void UpdateRaftProgress();
  // 对单个实例执行再平衡；内部完成 读表 -> 决策 -> 改映射 -> 写回
  void RebalanceInstance(uint64_t instance_id);
  // 请求 leader 把指定节点加入 Raft 组
  auto JoinRaftGroup(const std::string &leader_url, uint64_t node_id, const std::string &endpoint) -> bool;

 private:
  EtcdHttpClient etcd_client_;
  std::map<std::string, int> node_error_counts_;  // 错误计数器

  // 进行中的迁移任务（key 形如 "{instanceId}:{partitionId}"）。
  // 迁移涉及 etcd 写入与跨节点通知，非瞬时完成，需要防止 10 秒循环重复触发同一迁移。
  std::mutex migration_mutex_;
  std::set<std::string> active_migrations_;
};

// 允许落后 leader 的日志条数。超过该值视为尚未追平，不允许接管分区，
// 避免把流量切到数据不全的新节点上。
constexpr uint64_t kMaxLogLagForReady = 10;
}  // namespace vectordb
