# Raft 快照性能优化与 Learner 自动转正 —— 实施方案

状态：**进行中**（参数调整已完成，其余待实施）
日期：2026-09-24
关联：`vectordb-garden-raft-修复方案`（已完成，未编译验证）

---

## 一、背景

上一轮实现了 Raft 快照与日志压缩（`create_snapshot` 落索引并回调、快照流式传输三接口、learner 加入支持）。这套机制一旦真正生效，暴露出三个此前被掩盖的可用性问题；同时 learner 默认加入引入了一个 P0 回归。

| # | 问题 | 严重度 | 根因位置 |
|---|---|---|---|
| A | 快照制造阻塞 commit 线程，写入超时 | P1 | `raft_server.cxx:296`，commit 线程同步执行 `create_snapshot` |
| B | 每次快照全量写 10 种索引，停摆十几秒 | P1 | `collection.h:34-48` 预建 10 种索引 + `index_factory.cpp:63-88` 全量遍历 |
| C | 快照传输阻塞唯一 Raft 网络线程，leader 被推翻 | **P0** | `raft_stuff.cpp:26` `thread_pool_size_=1` + NuRaft 默认 `use_bg_thread_for_snapshot_io_=false` |
| D | 扩容后投票成员退化为 1，故障切换失效 | **P0** | `admin_service_impl.cpp:123` `as_learner` 默认 true + 无自动转正 |
| E | `GetTotalCount()` 恒返回 0，再平衡失效 | P1 | `index_factory.cpp:117-128` 硬编码 FLAT 选路 |

**问题 D 的详细后果**（已确认，NuRaft 源码佐证）：

- `handle_timeout.cxx:280` `if (!im_learner_)` → learner **不发起选举**
- `raft_server.cxx:596-600` `get_num_voting_members()` 排除 learner → **不计入法定人数**
- 按 `DOCKER_CLUSTER_REPORT.md` 既有步骤 AddFollower node-2/3（不传 `learner` 字段 → 默认 true）→ 投票成员从 3 个退化为 1 个 → **leader 挂掉后无人能接任**
- `master_service_impl.cpp:748-766` 的 `ready_` 判定只看 `alive && HasCaughtUp`，不检查 learner → learner 会被判定 ready 并接管分区读流量，却永远无法补位

---

## 二、已完成

### 2.1 本轮：参数调整（todo: tune-raft-params）✅

`src/cluster/raft_stuff.cpp`，三处，每处附取值理由注释：

| 参数 | 原值 | 新值 | 理由 |
|---|---|---|---|
| `asio_opt.thread_pool_size_` | 1 | **4** | 该线程池承载全部 Raft RPC（心跳/日志复制/快照传输/投票），单线程时任一 RPC 变慢即心跳停摆 |
| `params.use_bg_thread_for_snapshot_io_` | false（NuRaft 默认） | **true** | false 时 `read_logical_snp_obj` 由 worker 线程同步整文件读入内存，期间无法处理心跳。NuRaft 标注 Experimental，但默认组合风险更严重（用户已拍板用官方开关） |
| `params.client_req_timeout_` | 10000 | **60000** | 快照在 commit 线程同步执行，期间 `sm_commit_index_` 不推进，blocking `append_entries` 会一直等；10 秒兜不住 GB 级落盘 |

### 2.2 上一轮：快照与 learner 基础设施（已就位，待编译验证）

- `LogStateMachine::create_snapshot` 落索引并调用 `when_done`（此前空实现不调回调 → `compact()` 永不触发）
- `params.snapshot_distance_` 5 → **100000**，`reserved_log_items_` 5 → **1000**
- `Persistence::TakeSnapshot(uint64_t snapshot_log_id)` 重载，记录 Raft 日志位点
- 快照流式传输：`read_logical_snp_obj` / `save_logical_snp_obj` / `apply_snapshot`
- `AddSrv(..., as_learner)`、`PromoteLearner`、`IsVotingMember`、`GetPeerLastLogIdx`、`GetCommittedLogIdx`
- `AdminService/PromoteLearner` RPC（含追平校验：`GetPeerLastLogIdx + 1 >= GetCommittedLogIdx`）

---

## 三、待实施

### 3.1 `used_types_` 追踪与持久化（todo: track-used-index-types）

**位置**：`src/include/index/index_factory.h`、`src/index/index_factory.cpp`

**接口**：

```cpp
class IndexFactory {
public:
    // 标记某个索引类型已被写入过。快照只落盘被标记过的类型。
    void MarkUsed(IndexType type);
    // 供 SaveIndex/LoadIndex 内部使用
    auto GetUsedTypes() const -> std::set<IndexType>;
private:
    std::map<IndexType, void*> index_map_;
    std::set<IndexType> used_types_;
};
```

**SaveIndex 逻辑**（`index_factory.cpp:63-88` 改造）：

1. 遍历 `used_types_`（而非 `index_map_` 全量），按既有 if-else 链分发 Save
2. `FILTER` 无条件写入——GARDEN 搜索依赖它构造 bitmap，标量过滤走它
3. 写 sidecar 文件 `{folder_path}used_types.bin`，格式：`[uint32 count][uint32 type_value] × count`（二进制小端）

**LoadIndex 逻辑**（`index_factory.cpp:91-115` 改造）：

1. 读 sidecar 文件恢复 `used_types_`
2. **sidecar 不存在 → 兜底为"全部类型"**（旧快照兼容，否则旧数据升级后漏加载）
3. 只加载 `used_types_` 中的类型

### 3.2 写入侧标记（todo: mark-used-on-write）

**位置**：`src/database/vector_database.cpp`

- `Upsert`（`:141-280`）：switch 各分支写入成功后 `coll->index_factory.MarkUsed(index_type)`
- `BatchUpsert`（`:295+`）：同上
- FilterIndex 更新处（`:247+`）：无条件 `MarkUsed(IndexType::FILTER)`

**约束**：调用点必须覆盖**所有**写入路径，漏一处即漏写该索引。

### 3.3 `GetTotalCount()` 修复（并入本项）

**现状**（`index_factory.cpp:117-128`）：硬编码 FLAT → HNSW 选路。因 `InitAllIndices()` 必建 FLAT 且非 null，第一个分支必然命中，"退化到 HNSW"是死代码。业务用 HNSW/GARDEN 时**恒返回 0**。

**影响**：`admin_service_impl.cpp:308-315` 用它累加 `vector_count` 上报 → `NodeLoad::vector_count_` → `PlanMigration` 负载输入 → **再平衡决策失去依据**。

**修法**：改为遍历 `used_types_`（跳过无向量计数的 `FILTER`），返回首个非零计数；`used_types_` 为空时兜底为旧的 FLAT → HNSW 行为。约 15 行，由 3.5 的测试覆盖。

### 3.4 learner 状态上报（todo: report-learner-status）

打通"Master 知道谁是 learner"这条数据链：

| # | 位置 | 改动 |
|---|---|---|
| 1 | `raft_stuff.cpp:295` `GetAllNodesInfo()` | 返回值从 5 元组 `(id, endpoint, state, last_log_idx, last_succ_resp_us)` 扩展，增加 `bool is_learner`（来源：`srv_config::is_learner()`）。该函数仅被 `ListNode` 调用，可安全扩展 |
| 2 | `admin_service_impl.cpp:253+` `ListNode` | 输出 JSON 增加 `"learner": true/false` 字段 |
| 3 | `master_service_impl.cpp:444-498` `UpdateRaftProgress` | 解析 ListNode 响应中的 learner 字段，写回 etcd 节点表（**仅在变化时写入**，与 `lastLogIdx` 同策略，`:487-491`） |

### 3.5 Master 驱动自动转正（todo: auto-promote-learner）

**位置**：`src/httpserver/master_service_impl.cpp` `RebalanceInstance`（`:704+`）

**插入点**：读取节点表并算完 `ready_` 之后、`PlanMigration` **之前**——使再平衡决策基于最新成员状态。

**逻辑**（伪代码）：

```
// 在算完 ready_ 之后、PlanMigration 之前
for (const auto& load : node_loads) {
    if (!load.is_learner)  continue;   // 字段缺失按非 learner（用户拍板 #3）
    if (!load.ready_)      continue;   // 未追平不转正
    if (load.node_id_ == leader_node_id) continue;  // 防御：leader 不可能是 learner

    HTTP POST {leader_url}/AdminService/PromoteLearner
    body: {"nodeId": load.node_id_}

    成功 → info 日志
    失败 → warn 日志，不中断（下一轮 10 秒后自然重试）
}
```

**关键性质**：

- **leader 侧已有实时把关**（`admin_service_impl.cpp:171-181` 比对 `GetPeerLastLogIdx` 与 `GetCommittedLogIdx`），Master 基于 10 秒前 etcd 快照的粗筛不会造成错误转正，最坏情况是被 leader 拒绝后下轮重试
- **转正 = 一条 Raft conf 日志**，频率极低（每个节点一次），无性能顾虑
- 转正后 `get_num_voting_members()` +1，法定人数提高；因节点已追平，新写入不会被卡住

**已拍板的设计决策**：

| 决策点 | 结论 | 依据 |
|---|---|---|
| 谁驱动 | **Master**（非 leader 扫描） | `HasCaughtUp`/`ready_` 逻辑已存在，复用不重写；单一决策点无并发竞态；`lastLogIdx` 数据现成 |
| 周期 | **复用 10 秒**（与 UpdateNodeStates/RunBalancer 同频） | 加节点本身是分钟级操作，10 秒延迟无感 |
| Master 挂掉 | **不加节点侧兜底** | 已转正节点照常参与选举，故障切换不受影响；只是新扩容节点暂不转正，不会比现状更糟 |
| learner 字段缺失 | **按非 learner，不转正** | 避免"老集群全节点被误判为 learner 反复转正"产生垃圾日志 |

### 3.6 可观测性与测试（todo: add-observability-and-tests）

**日志**：

- `Persistence::TakeSnapshot`（`persistence.cpp:159-174`）：开始/结束各一条 `info`，带耗时（`std::chrono::steady_clock`）与落盘字节数——这是观测"快照停摆时长"的直接手段
- `MarkUsed`：仅首次标记时打 `debug`，避免高频刷屏
- Master 转正：成功 `info` / 失败 `warn`，不得静默
- 快照失败：`error`，不得静默

**测试**：新增 `test/index/index_factory_used_types_test.cpp`（CMake 按 `test/*/*test.cpp` 自动 glob）：

1. 标记后只写这些类型（检查目录文件数）
2. `used_types.bin` 持久化往返
3. 旧快照（无 sidecar）兜底为全类型
4. `GetTotalCount()` 按 `used_types_` 选路（业务用 HNSW 时不再恒 0）

---

## 四、正确性红线（实施时逐条自查）

| # | 红线 | 违反后果 |
|---|---|---|
| 1 | `used_types_` 必须持久化 | 重启丢失 → 快照漏写 → 而 `compact()` 已删对应日志 → **数据永久丢失** |
| 2 | `FilterIndex` 视为始终使用 | GARDEN 搜索与标量过滤依赖它构造 bitmap |
| 3 | `LoadIndex` 对旧快照（无 sidecar）兜底为全类型 | 旧数据升级后漏加载 |
| 4 | 转正步骤必须在 `PlanMigration` 之前 | 再平衡基于过期成员状态决策 |
| 5 | `PromoteLearner` 前置校验不得移除 | 未追平就转正 → 法定人数提高 → 新写入被卡住直至超时 |

---

## 五、验收标准

| # | 验证项 | 方法 |
|---|---|---|
| 1 | 编译通过 | Linux 环境 `third_party/build.sh` + `cmake --build`（**最高优先级**，此前所有改动均未编译验证） |
| 2 | used_types 单测通过 | `./test/index_factory_used_types_test` |
| 3 | 既有 GARDEN 测试不回归 | `./test/garden_persistence_test` 等 |
| 4 | 日志压缩生效 | `snapshot_distance_` 临时调 50，写几百条，观察 `logs_.size()` 回落、`snap/` 出现文件 |
| 5 | 快照期间写入不超时 | 压测观察快照触发时写入 P99 与失败数 |
| 6 | 快照传输不推翻 leader | 三节点，让一个节点长期落后触发快照传输，观察 `ListNode` 角色变化 |
| 7 | **故障切换不回归（D 项关键验收）** | 按 `DOCKER_CLUSTER_REPORT.md` 步骤 AddFollower ×2 → 等 ~10 秒 → `ListNode` 确认 node-2/3 的 `learner=false` → 停 leader → 确认新 leader 接任 |
| 8 | 再平衡恢复 | 业务用 HNSW 写入后，`GetNodeInfo` 上报的 `vectorCount` 非 0 |

---

## 六、明确不在本次范围

| 项 | 原因 |
|---|---|
| 快照对象分块（大文件切分） | 用户已确认延后。当前风险：GARDEN 的 `_garden.bin` 过大时单个 object 可能超出 NuRaft 传输上限 |
| 向量维度校验 | 已确认存在内存越界风险（`Upsert` 不比对 `meta.dimension`，`BatchUpsert` 混合维度会错位整批数据），**独立成项待排期** |
| Master 节点侧兜底 | 用户已拍板不加 |
| `insert`/`upsert` 语义分离 | 当前两者等价（都是覆盖写）；如需"重复 id 报错"语义需另起 |
| HNSW 单条插入无容量扩容 | `InsertVectors`（单条）不做 `resizeIndex`，超 `num_data` 可能异常。与维度校验一并排期 |

---

## 七、执行顺序

```
[✅] 1. 参数调整            tune-raft-params
[ ] 2. used_types_ + sidecar + GetTotalCount    track-used-index-types
[ ] 3. 写入侧 MarkUsed      mark-used-on-write        （依赖 2）
[ ] 4. learner 状态上报     report-learner-status
[ ] 5. Master 自动转正      auto-promote-learner      （依赖 4）
[ ] 6. 日志 + 单测          add-observability-and-tests（依赖 3）
[ ] 7. 验证 + review        verify-and-review          （依赖 1/5/6，需 Linux）
```

依赖关系上，**2/3 与 4/5 两条线互相独立**，可并行推进；7 必须等全部完成后在 Linux 环境做。
