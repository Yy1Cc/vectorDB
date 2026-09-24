#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace vectordb {

// 节点的负载快照。由 Master 从 etcd 中的节点表里组装而来。
//
// 注意：partition_count_ 仅作为调试/展示用的参考值，
// PlanMigration 的决策一律依据传入的分区表实时统计，
// 避免"节点表里缓存的分区数"与"分区表"两个数据源不一致。
struct NodeLoad {
  uint64_t node_id_{0};
  uint64_t vector_count_{0};      // 节点上报：向量总条数
  uint64_t disk_usage_bytes_{0};  // 节点上报：RocksDB + WAL + 快照的磁盘占用
  int partition_count_{0};        // 参考值，不参与决策
  bool ready_{true};              // 是否已完成 Raft 追平、可接管分区
};

// 一次迁移计划：把 partition_id_ 的归属从 from_node_id_ 改为 to_node_id_
struct MigrationPlan {
  uint64_t partition_id_{0};
  uint64_t from_node_id_{0};
  uint64_t to_node_id_{0};
};

// 允许的最大分区数差距。差距 <= 该值即视为已均衡，不再迁移。
// 取 1 而非 0：可避免在偶数分区场景下两个节点来回互迁造成震荡。
constexpr int kDefaultImbalanceThreshold = 1;

// 再平衡决策（纯函数，不触碰 etcd / Raft / 网络，可脱离集群单测）。
//
// 策略："分区数最少优先" —— 当最忙与最闲节点的分区数差距超过阈值时，
// 把最忙节点上的一个分区迁给最闲节点。一次只规划一个迁移，由调用方限流。
//
// partition_to_node: 分区表，每一项为 (partition_id, owning_node_id)
// node_loads:        各节点负载快照，仅 ready_ 为 true 的节点参与调度
// threshold:         允许的分区数差距上限（默认 kDefaultImbalanceThreshold）
//
// 返回 std::nullopt 表示当前已均衡或无法迁移（节点不足、无可用节点等）。
// 对同一输入必定返回同一结果（按 node_id / partition_id 升序做确定性选择）。
auto PlanMigration(const std::vector<std::pair<uint64_t, uint64_t>>& partition_to_node,
                   const std::vector<NodeLoad>& node_loads,
                   int imbalance_threshold = kDefaultImbalanceThreshold) -> std::optional<MigrationPlan>;

}  // namespace vectordb
