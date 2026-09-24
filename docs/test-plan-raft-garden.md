# GARDEN + Raft 修复链路 —— 测试计划

状态：**待执行**（所有改动均未编译验证）
日期：2026-09-24
配套文档：`docs/raft-snapshot-learner-promotion.md`

---

## 〇、测试前提（顺序不可颠倒）

| # | 前提 | 说明 |
|---|---|---|
| P1 | **编译通过** | `third_party/build.sh`（Linux）+ `cmake --build`。这是第一道关卡，当前所有改动都未过编译 |
| P2 | 构建类型确认 | `CMakeLists.txt` 的 `set(CMAKE_BUILD_TYPE Debug)` 会覆盖 Dockerfile 的 Release，压测前必须确认 |
| P3 | 干净的测试目录 | 清空 `wal_path` / `snap_path`，避免旧快照干扰 |
| P4 | 三节点 docker-compose 可起 | etcd + master 6060 + proxy 6061 + vdb-node-1/2/3 |

**风险提示**：改动横跨持久化格式（GARDEN 单文件容器）、NuRaft 参数、HTTP 响应语义三类。
**旧快照与旧 WAL 不兼容新格式**——测试必须从空目录开始，生产升级需另行迁移方案（见第六节）。

---

## 一、单元测试（编译通过后可直接跑）

### 1.1 `garden_persistence_test`（新增）

覆盖 GARDEN 持久化与过滤正确性，对应修复：

| 用例 | 验证的修复 |
|---|---|
| `SaveLoadRoundTrip` | `IndexFactory::SaveIndex` 补 GARDEN 分支 + 单文件容器格式 |
| `SnapshotIsSingleFile` | 一值一文件 → 单文件（旧格式 3 离散值+10 桶=14 个文件，新格式 1 个） |
| `ContinuousRangeFilterHitsCorrectBuckets` | `has_range` 区间过滤命中正确桶 |
| `ContinuousFilterWithoutRangeStillReturnsResults` | `has_range` 只写不读导致等值过滤退化成只搜第 0 号桶 |
| `DeletedVectorsStayDeletedAfterReload` | 删除后落盘，重启不"复活" |

```bash
./build/test/index_factory_used_types_test 2>/dev/null || true
./build/test/garden_persistence_test
```

**判定标准**：全部 PASS。其中 `ContinuousFilterWithoutRangeStillReturnsResults` 在修复前必然失败（返回空结果），是 `has_range` 修复的直接回归断言。

### 1.2 `index_factory_used_types_test`（新增）

覆盖快照剪枝与 `GetTotalCount` 修复：

| 用例 | 验证的修复 |
|---|---|
| `SaveOnlyMarkedTypes` | 未写入的索引类型不落盘（目录只有 3 个文件而非 11 个） |
| `SidecarRoundTrip` | `used_types.bin` 持久化往返正确 |
| `LegacySnapshotFallbackAndPrune` | 旧快照（无 sidecar）兜底为全类型，加载后剪枝空类型并固化 |
| `GetTotalCountRoutesByUsedTypes` | 业务用 HNSW 时 `GetTotalCount()` 返回 7 而非 0 |

**判定标准**：全部 PASS。

### 1.3 既有 GARDEN 测试回归

```bash
./build/test/garden_index_test
./build/test/garden_continuous_field_test 2>/dev/null || true
```

**判定标准**：不因 `has_range` 修复与持久化格式变更而回归。

---

## 二、单节点功能验证（Linux，非集群）

以下每项都给出操作步骤、预期结果与失败时的含义。

### 2.1 GARDEN 快照落盘与重启恢复（验证修复：SaveIndex GARDEN 分支）

```
1. 启动单节点
2. upsert 若干条（含 string/int 标量字段，indexType=GARDEN_HNSW）
3. 注册 GARDEN 字段：registerGardenField（discrete + continuous 各一个）
4. curl -X POST http://localhost:7781/AdminService/snapshot
5. ls $SNAP_PATH          ← 应看到 {coll}_GARDEN_HNSW...garden.bin（单文件）+ MaxLogID + used_types.bin
6. 记录文件大小
7. kill -9 进程 → 重启
8. search 同一批向量
```

| 预期 | 失败时含义 |
|---|---|
| 步骤 5 出现 `.garden.bin` 单文件 | SaveIndex 分支没生效 |
| 步骤 8 返回正确 top-k 且距离一致 | LoadIndex/容器格式解析有问题 |
| 日志出现 `GardenIndex: loaded from ...` | 加载路径未走到 |

### 2.2 MaxLogID 生效：重启不再全量重放（验证修复：MaxLogID 三处 bug）

```
1. upsert 200 条 → 手动 snapshot
2. 再 upsert 50 条（不 snapshot）
3. 记录 WAL 行数 = 250
4. kill -9 → 重启，观察日志
```

| 预期 | 失败时含义 |
|---|---|
| 日志出现 `Snapshot MaxLogID file not found` 仅在首次；重启后应读到 `Loading snapshot Max log ID 200` | 文件名仍不一致 |
| 重启只重放 50 条（日志里 `No more WAL log entries to read` 出现得早） | `last_snapshot_id_` 仍为 0，快照零收益 |
| 重启后 search 全部正确 | MaxLogID 与索引不一致 |

### 2.3 upsert 失败返回错误（验证修复：AppendEntries 返回 bool）

```
1. 对 follower 节点直接发 upsert（proxy 关闭，直连 7782）
```

| 预期 | 失败时含义 |
|---|---|
| HTTP 返回非 0 retCode + "not the leader" | 返回值没传上来 |
| 日志有 `Cannot append entries: current node (id=2) is not the leader` | — |

### 2.4 insert 走 Raft 且对 GARDEN 生效（验证修复：insert 接 Raft）

```
1. leader 上 POST /UserService/insert（indexType=GARDEN_HNSW）
2. 立即 search 该 id
3. follower 上 search 该 id（直连 7782）
```

| 预期 | 失败时含义 |
|---|---|
| 两侧都能搜到 | insert 未走 Raft（只在本节点） |
| GARDEN 也能 insert（不再是静默 no-op） | 重放侧没覆盖 |

### 2.5 连续字段等值过滤召回（验证修复：has_range）

```
1. 注册连续字段 price，range=[0,10000]，bucket=1000
2. 插入 5000 条，price = i*2 均匀分布
3. 过滤 price == 5000（EQUAL 操作）查询，k=10
```

| 预期 | 失败时含义 |
|---|---|
| 返回 price=5000 附近的结果，非空 | 仍退化成只搜第 0 号桶 |
| 与不过滤的暴力结果召回一致 | — |

### 2.6 快照自动触发 + WAL 截断（验证修复：create_snapshot + compact）

**先把 `snapshot_distance_` 临时从 100000 调到 50**（否则要写 10 万条才触发）。

```
1. upsert 300 条
2. 观察日志：出现 "Created snapshot at log idx N" 和 "TakeSnapshot end ... ms total"
3. 确认 $SNAP_PATH 出现快照文件
4. 确认 WAL 文件行数明显减少（TruncateWalBefore 生效）
5. 确认进程 RSS 不再随写入单调增长
6. 测完把 snapshot_distance_ 改回 100000
```

| 预期 | 失败时含义 |
|---|---|
| 出现 snapshot 日志 | `when_done` 未被调用或 `TakeSnapshot` 抛异常 |
| `logs_.size()` 回落 | `compact_async` 未触发 |
| WAL 行数减少 | `TruncateWalBefore` 未生效或 ParseWalLogId 失败（会保守保留全部） |

**这是本计划最关键的一项**——它同时验证了内存 OOM 修复与磁盘 WAL 无限增长修复。

### 2.7 快照耗时观测（验证修复：TakeSnapshot 日志）

在 2.6 的日志里看 `TakeSnapshot: collection 'x' saved in N ms`。

| 预期 | 含义 |
|---|---|
| used_types_ 剪枝生效后，耗时显著低于全量（约 1/10） | 剪枝未生效，或 used_types_ 被错误地置为全量 |
| 总耗时 < `client_req_timeout_`(60s) | 否则快照期间写入会超时 |

### 2.8 GetTotalCount / 负载上报（验证修复：GetTotalCount）

```
1. 业务只用 HNSW 写入若干条
2. curl http://localhost:7781/AdminService/GetNodeInfo（或 ListNode/上报接口）
```

| 预期 | 失败时含义 |
|---|---|
| `vectorCount` 反映 HNSW 条数（非 0） | 仍硬编码读 FLAT |

---

## 三、集群行为验证（三节点 docker-compose）

### 3.1 集群组建与数据复制回归

按 `DOCKER_CLUSTER_REPORT.md` 原有流程回归一遍（组建、复制、查询、故障切换）。
**新增关注点**：`ListNode` 输出里应出现 `learner` 字段。

### 3.2 learner 默认加入 + 自动转正（验证修复：P0 回归）

```
1. node-1 单独运行，写入 1000 条
2. AddFollower node-2（不传 learner 字段 → 默认 learner）
3. 10 秒内轮询 ListNode：
   - 前几次 learner=true
   - 追平后应看到 learner=false
4. Master 日志应出现 "Auto-promoting learner node 2"
5. AddFollower node-3，同样验证
6. 停掉 node-1（当前 leader）
7. 10 秒内确认 node-2 或 node-3 成为新 leader
8. 写入新数据，确认全部节点一致
```

| 预期 | 失败时含义 |
|---|---|
| 步骤 3 learner 由 true 变 false | 自动转正链路断了（etcd 字段 / Master 判定 / RPC 任一环节） |
| **步骤 7 有节点接任** | learner 未转正 → 投票成员只有 node-1 → 集群不可选主（**P0 回归复现**） |
| 步骤 8 三节点一致 | 转正后新写入未同步 |

**这是本计划第二关键的一项**，直接验证"扩容后故障切换能力"是否恢复。

### 3.3 元数据变更走 Raft（验证修复：registerGardenField / createCollection）

```
1. 三节点集群，leader 上 createCollection（dim=8）
2. follower 上 getCollectionInfo 同名集合   ← 应存在（此前不存在）
3. leader 上 registerGardenField（continuous，price [0,10000] bucket 1000）
4. 在 follower 上对同一字段做 GARDEN 过滤查询
5. 重复 registerGardenField 同一字段（幂等性）
```

| 预期 | 失败时含义 |
|---|---|
| 步骤 2 副本上有该集合 | createCollection 未走 Raft |
| 步骤 4 follower 走子图路径（日志可见 GardenIndex 分支）而非全量图 | 连续字段注册未复制，副本路由退化 |
| 步骤 5 不重建分桶、不丢数据 | 幂等保护缺失（重复 Register 会清空桶） |

### 3.4 落后节点走快照传输（验证修复：snapshot 流式传输三接口）

**前置**：先把 `snapshot_distance_` 调小让 compact 真正发生（否则日志不会被删，走不到快照传输分支）。

```
1. node-1 写入大量数据，触发至少一次快照与 compact
2. AddFollower node-2（作为 learner）
3. 观察 node-2 日志：出现 save_logical_snp_obj / apply_snapshot
4. 确认 node-2 追平后数据正确
5. 确认 node-1 全程未被推翻（ListNode role 一直是 leader）
```

| 预期 | 失败时含义 |
|---|---|
| node-2 日志出现快照传输 | `read_logical_snp_obj` 仍返回 0（空实现残留） |
| node-1 role 保持 leader | **快照传输期间心跳停摆 → 误选举**（`use_bg_thread_for_snapshot_io_` / `thread_pool_size_` 未生效） |
| node-2 数据与 node-1 一致 | apply_snapshot 未正确加载 |

### 3.5 集群配置持久化（验证修复：save_config/save_state）

```
1. 三节点集群正常运行
2. kill -9 全部节点 → 依次重启（不执行 AddFollower）
```

| 预期 | 失败时含义 |
|---|---|
| 无需手动 AddFollower，集群自动恢复成员关系 | save_config 未落盘或路径错误 |
| 日志出现 `restored cluster config from ... (N servers)` | — |
| term 不归零（日志 `restored raft state (term=N)`，N>0） | save_state 未落盘，选举安全性受损 |

### 3.6 冲突日志回滚（验证修复：write_at 回滚 WAL）

**难直接构造**，需要人为制造脑裂场景：

```
1. 三节点，写若干条
2. 隔离 node-3（docker network disconnect）
3. node-1/2 继续写入，产生新 leader 任期
4. 恢复 node-3 → 它会被要求 write_at 截断冲突日志
5. 检查 node-3 的 WAL 文件：不应包含被 Raft 否决的条目
6. 重启 node-3 → 数据应与 node-1/2 一致
```

| 预期 | 失败时含义 |
|---|---|
| node-3 WAL 行数在截断后减少 | write_at 仍不回滚 WAL |
| 重启后 node-3 数据与多数派一致 | 未提交日志被错误重放 |

---

## 四、性能验证

### 4.1 快照期间写入 P99（验证参数：thread_pool_size_/use_bg_thread_for_snapshot_io_/client_req_timeout_）

```
1. snapshot_distance_ 调小到 1000
2. 压测写入（如 wrk / 自带 bench），持续触发快照
3. 记录快照触发前后的 P99 与失败率
4. 对照组：改回原参数 thread_pool_size_=1 + 不开 bg snapshot IO
```

| 预期 | 含义 |
|---|---|
| 新参数下快照期间无秒级尖刺、无超时 | 解耦生效 |
| 对照组出现秒级尖刺或超时 | 证明旧配置确有问题（也验证测试方法本身有效） |

### 4.2 GARDEN 分区下不退化（验证修复：kBruteBound 相对化）

```
1. 单节点：写入 5 万条，过滤查询记录 QPS 与路由分支日志
2. 三节点（数据分摊）：同样负载
3. 对比日志里的路由分支
```

| 预期 | 失败时含义 |
|---|---|
| 分区下仍大量走子图路径（日志可见），而非全部暴力 | 相对阈值未生效 |
| 分区下 QPS 与单机同量级 | 退化为暴力 |

### 4.3 proxy 过滤请求广播（验证修复：partitionKey 冲突）

```
1. 配置 partitionKey，写入数据使不同分区持有不同 brand
2. 发送带 filter（brand 条件）的读请求
3. 检查结果完整性（对照单机全量结果）
```

| 预期 | 失败时含义 |
|---|---|
| 结果包含所有分区中满足条件的数据（日志出现 broadcasting） | 仍按 partitionKey 单路由，结果静默不全 |

---

## 五、验收清单（打勾表）

| # | 项目 | 阶段 | 结果 |
|---|---|---|---|
| 1 | 编译通过 | 前提 | ☐ |
| 2 | `garden_persistence_test` 全过 | 单测 | ☐ |
| 3 | `index_factory_used_types_test` 全过 | 单测 | ☐ |
| 4 | 既有 GARDEN 测试不回归 | 单测 | ☐ |
| 5 | GARDEN 快照落盘 + 重启恢复 | 单节点 | ☐ |
| 6 | MaxLogID 生效，重启不全量重放 | 单节点 | ☐ |
| 7 | upsert 失败返回错误 | 单节点 | ☐ |
| 8 | insert 走 Raft 且 GARDEN 生效 | 单节点 | ☐ |
| 9 | 连续字段等值过滤召回正常 | 单节点 | ☐ |
| 10 | 快照自动触发 + logs_ 回落 + WAL 截断 | 单节点 | ☐ |
| 11 | vectorCount 上报非 0 | 单节点 | ☐ |
| 12 | 集群组建/复制/查询回归 | 集群 | ☐ |
| 13 | learner 自动转正（10s 内） | 集群 | ☐ |
| 14 | **leader 挂掉有节点接任** | 集群 | ☐ |
| 15 | createCollection / registerGardenField 副本一致 | 集群 | ☐ |
| 16 | 落后节点快照传输 + leader 不被推翻 | 集群 | ☐ |
| 17 | 重启后集群成员自动恢复、term 保留 | 集群 | ☐ |
| 18 | 冲突日志回滚 | 集群 | ☐ |
| 19 | 快照期间写入无超时 | 性能 | ☐ |
| 20 | 分区下 GARDEN 不退化 | 性能 | ☐ |

**最低可发布标准**：1 + 2 + 3 + 5 + 6 + 10 + 13 + 14 全过。
**14 是 P0 回归的直接验证**，若失败则 learner 自动转正链路有断点，不允许上线。

---

## 六、已知风险与迁移注意

| 风险 | 说明 | 缓解 |
|---|---|---|
| **持久化格式不兼容** | GARDEN 单文件容器格式 + `used_types.bin` + 新的 MaxLogID 文件名，与旧快照/WAL 不兼容 | 测试从空目录开始；生产升级需先导出再导入，或接受一次全量重放 |
| `use_bg_thread_for_snapshot_io_` 为 Experimental | NuRaft 标注实验性，行为可能随版本变化 | 三节点验证项 16 专门覆盖 |
| `snapshot_distance_=100000` 下的验证盲区 | 默认参数下单测很难触发快照 | 测试时临时调小；上线前务必改回 |
| `write_at` 回滚与 compact 的交互 | `write_at` 会把 WAL 截到 index-1，若此时已有快照覆盖更远位置，可能截掉已快照数据 | 验证项 3.6 专门覆盖；若出现数据不一致，需在 `write_at` 里加 `index > last_snapshot_id_` 保护 |
| 快照 object 过大 | GARDEN `_garden.bin` 可能超过 NuRaft 单次传输上限 | 已知限制，本期不做分块；验证项 16 若失败优先怀疑此处 |
| 维度未校验 | `Upsert` 不比对 `meta.dimension`，混维度插入会内存越界 | **测试时严禁发送维度不一致的请求**；该问题已知待修 |
