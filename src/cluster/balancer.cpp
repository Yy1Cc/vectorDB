#include "cluster/balancer.h"

#include <algorithm>
#include <map>

namespace vectordb {

auto PlanMigration(const std::vector<std::pair<uint64_t, uint64_t>>& partition_to_node,
                   const std::vector<NodeLoad>& node_loads,
                   int imbalance_threshold) -> std::optional<MigrationPlan> {
  // 无分区可搬、节点不足两个、或阈值非法时，没有迁移的意义
  if (partition_to_node.empty() || node_loads.size() < 2 || imbalance_threshold < 0) {
    return std::nullopt;
  }

  // 只有 ready 的节点参与调度：未追平的节点不能接管分区，
  // 也不从其上迁出（其数据尚未完整，迁出可能造成分区数据缺失）
  std::vector<const NodeLoad*> candidates;
  candidates.reserve(node_loads.size());
  for (const auto& load : node_loads) {
    if (load.ready_) {
      candidates.push_back(&load);
    }
  }
  if (candidates.size() < 2) {
    return std::nullopt;
  }

  // 以分区表为唯一事实来源统计各节点持有的分区数
  std::map<uint64_t, int> partition_counts;
  for (const auto* load : candidates) {
    partition_counts[load->node_id_] = 0;
  }
  for (const auto& [partition_id, node_id] : partition_to_node) {
    auto it = partition_counts.find(node_id);
    if (it != partition_counts.end()) {
      it->second++;
    }
    // 分区挂在未 ready 或已下线的节点上：本轮不处理，交由缩容/修复流程处理
  }

  // 按 node_id 升序排列后再取极值，保证同一输入得到同一结果（幂等、可测）
  std::vector<const NodeLoad*> ordered = candidates;
  std::sort(ordered.begin(), ordered.end(),
            [](const NodeLoad* a, const NodeLoad* b) { return a->node_id_ < b->node_id_; });

  auto count_of = [&partition_counts](const NodeLoad* load) {
    auto it = partition_counts.find(load->node_id_);
    return it == partition_counts.end() ? 0 : it->second;
  };

  const NodeLoad* busiest = nullptr;
  const NodeLoad* idlest = nullptr;
  for (const auto* load : ordered) {
    if (busiest == nullptr || count_of(load) > count_of(busiest)) {
      busiest = load;
    }
    if (idlest == nullptr || count_of(load) < count_of(idlest)) {
      idlest = load;
    }
  }
  if (busiest == nullptr || idlest == nullptr) {
    return std::nullopt;
  }

  // 已均衡：差距未超过阈值则不迁移（阈值 1 时可容忍 1 个分区的差距，避免震荡）。
  // 全部节点持有时该差值自然为 0，同样在此收敛。
  if (count_of(busiest) - count_of(idlest) <= imbalance_threshold) {
    return std::nullopt;
  }
  if (busiest->node_id_ == idlest->node_id_) {
    return std::nullopt;
  }

  // 从最忙节点上挑一个分区迁出。分区自身的负载无从拆分（只有节点级 vector_count），
  // 因此按 partition_id 升序取第一个，保证确定性。
  std::optional<uint64_t> partition_to_move;
  for (const auto& [partition_id, node_id] : partition_to_node) {
    if (node_id == busiest->node_id_) {
      if (!partition_to_move || partition_id < *partition_to_move) {
        partition_to_move = partition_id;
      }
    }
  }
  if (!partition_to_move) {
    // 最忙节点在分区表中没有任何分区（数据不一致），无从迁出
    return std::nullopt;
  }

  MigrationPlan plan;
  plan.partition_id_ = *partition_to_move;
  plan.from_node_id_ = busiest->node_id_;
  plan.to_node_id_ = idlest->node_id_;
  return plan;
}

}  // namespace vectordb
