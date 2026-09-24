#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "cluster/balancer.h"
#include "gtest/gtest.h"

namespace vectordb {

namespace {

// 分区表：每一项为 (partition_id, owning_node_id)
using PartitionTable = std::vector<std::pair<uint64_t, uint64_t>>;

auto MakeLoad(uint64_t node_id, bool ready = true, uint64_t vector_count = 0) -> NodeLoad {
  NodeLoad load;
  load.node_id_ = node_id;
  load.ready_ = ready;
  load.vector_count_ = vector_count;
  return load;
}

// 统计某节点在分区表中持有多少个分区，用于校验迁移前后的分布
auto CountPartitions(const PartitionTable& table, uint64_t node_id) -> int {
  int count = 0;
  for (const auto& [partition_id, owner] : table) {
    if (owner == node_id) {
      ++count;
    }
  }
  return count;
}

// 应用迁移计划，返回新的分区表
auto ApplyPlan(const PartitionTable& table, const MigrationPlan& plan) -> PartitionTable {
  PartitionTable updated = table;
  for (auto& [partition_id, owner] : updated) {
    if (partition_id == plan.partition_id_) {
      owner = plan.to_node_id_;
    }
  }
  return updated;
}

}  // namespace

// 已均衡（差距 <= 阈值）时不应产生迁移计划
TEST(BalancerTest, BalancedClusterNeedsNoMigration) {
  PartitionTable table = {{0, 1}, {1, 1}, {2, 2}, {3, 2}};  // N1=2, N2=2
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};

  auto plan = PlanMigration(table, loads);

  EXPECT_FALSE(plan.has_value());
}

// 差距超过阈值时，从最忙节点迁往最闲节点
TEST(BalancerTest, MigratesFromBusiestToIdlest) {
  PartitionTable table = {{0, 1}, {1, 1}, {2, 1}, {3, 2}};  // N1=3, N2=1, 差距 2
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};

  auto plan = PlanMigration(table, loads);

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->from_node_id_, static_cast<uint64_t>(1));
  EXPECT_EQ(plan->to_node_id_, static_cast<uint64_t>(2));
  // 同一节点上的多个分区按 partition_id 升序取第一个，保证确定性
  EXPECT_EQ(plan->partition_id_, static_cast<uint64_t>(0));
}

// 新节点加入（0 分区）时应从最忙节点接管一个分区
TEST(BalancerTest, NewNodeWithZeroPartitionsReceivesPartition) {
  PartitionTable table = {{0, 1}, {1, 1}};  // N1=2, N2=0, 差距 2
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};

  auto plan = PlanMigration(table, loads);

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->from_node_id_, static_cast<uint64_t>(1));
  EXPECT_EQ(plan->to_node_id_, static_cast<uint64_t>(2));
}

// 收敛性：反复应用迁移计划后应达到均衡并停止
TEST(BalancerTest, RepeatedMigrationConverges) {
  PartitionTable table = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 2}};  // N1=4, N2=1
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};

  int iterations = 0;
  const int kMaxIterations = 10;
  while (iterations < kMaxIterations) {
    auto plan = PlanMigration(table, loads);
    if (!plan.has_value()) {
      break;
    }
    table = ApplyPlan(table, *plan);
    ++iterations;
  }

  EXPECT_LT(iterations, kMaxIterations);  // 必须在有限步内收敛
  EXPECT_LE(CountPartitions(table, 1), 3);
  EXPECT_GE(CountPartitions(table, 2), 2);
  // 收敛后差距不应超过阈值
  EXPECT_LE(CountPartitions(table, 1) - CountPartitions(table, 2), kDefaultImbalanceThreshold);
}

// 只有一个节点时无从迁移
TEST(BalancerTest, SingleNodeNeverMigrates) {
  PartitionTable table = {{0, 1}, {1, 1}};
  std::vector<NodeLoad> loads = {MakeLoad(1)};

  auto plan = PlanMigration(table, loads);

  EXPECT_FALSE(plan.has_value());
}

// 空输入（无分区表或无节点）不产生计划
TEST(BalancerTest, EmptyInputYieldsNoPlan) {
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};
  EXPECT_FALSE(PlanMigration({}, loads).has_value());

  PartitionTable table = {{0, 1}, {1, 2}};
  EXPECT_FALSE(PlanMigration(table, {}).has_value());
  EXPECT_FALSE(PlanMigration({}, {}).has_value());
}

// 未 ready 的节点既不作为迁出方也不作为迁入方
TEST(BalancerTest, SkipsNodesThatAreNotReady) {
  PartitionTable table = {{0, 1}, {1, 1}, {2, 1}};  // N1=3
  // N2 未追平，N3 已就绪但当前 0 分区
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2, /*ready=*/false), MakeLoad(3)};

  auto plan = PlanMigration(table, loads);

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->from_node_id_, static_cast<uint64_t>(1));
  EXPECT_EQ(plan->to_node_id_, static_cast<uint64_t>(3));  // 跳过未 ready 的 N2
}

// 只有一个 ready 节点时无法迁移（其余节点均未就绪）
TEST(BalancerTest, OnlyOneReadyNodeYieldsNoPlan) {
  PartitionTable table = {{0, 1}, {1, 1}};
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2, false), MakeLoad(3, false)};

  auto plan = PlanMigration(table, loads);

  EXPECT_FALSE(plan.has_value());
}

// 节点数多于分区数时，多余节点保持 0 分区且不应来回震荡
TEST(BalancerTest, MoreNodesThanPartitionsStaysStable) {
  PartitionTable table = {{0, 1}};  // P=1, N=3: N1=1, N2=0, N3=0, 差距 1
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2), MakeLoad(3)};

  auto plan = PlanMigration(table, loads);

  // 差距 1 未超过默认阈值 1，不应迁移（否则会在节点间反复横跳）
  EXPECT_FALSE(plan.has_value());
}

// 阈值为 0 时，任何差距都会触发迁移
TEST(BalancerTest, ZeroThresholdMigratesOnAnyGap) {
  PartitionTable table = {{0, 1}, {1, 2}, {2, 2}};  // N1=1, N2=2, 差距 1
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};

  EXPECT_FALSE(PlanMigration(table, loads).has_value());  // 默认阈值下不迁移
  auto plan = PlanMigration(table, loads, 0);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->from_node_id_, static_cast<uint64_t>(2));
  EXPECT_EQ(plan->to_node_id_, static_cast<uint64_t>(1));
}

// 分区挂在未知（已下线）节点上时，不应把该分区误判为可迁出
TEST(BalancerTest, IgnoresPartitionsOwnedByUnknownNodes) {
  // 分区 9 挂在未在节点表中的节点 99 上
  PartitionTable table = {{0, 1}, {1, 1}, {2, 2}, {9, 99}};
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2)};  // N1=2, N2=1, 差距 1

  auto plan = PlanMigration(table, loads);

  // 差距 1 未超阈值，且孤儿分区不计入任何已知节点，不应产生迁移
  EXPECT_FALSE(plan.has_value());
}

// 幂等性：同一输入连续多次调用必须得到完全相同的结果
TEST(BalancerTest, PlanIsDeterministic) {
  PartitionTable table = {{0, 1}, {1, 1}, {2, 1}, {3, 2}, {4, 3}};
  std::vector<NodeLoad> loads = {MakeLoad(1), MakeLoad(2), MakeLoad(3)};

  auto first = PlanMigration(table, loads);
  auto second = PlanMigration(table, loads);
  auto third = PlanMigration(table, loads);

  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(first->partition_id_, second->partition_id_);
  EXPECT_EQ(first->from_node_id_, second->from_node_id_);
  EXPECT_EQ(first->to_node_id_, second->to_node_id_);
  EXPECT_EQ(second->partition_id_, third->partition_id_);
  EXPECT_EQ(second->from_node_id_, third->from_node_id_);
  EXPECT_EQ(second->to_node_id_, third->to_node_id_);
}

}  // namespace vectordb
