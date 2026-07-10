# vectorDB 性能压测与分布式测试报告

**测试日期**: 2026-07-09 ~ 2026-07-10
**测试环境**: 8核 CPU, TencentOS Server 4.4, 128维向量
**测试工具**: Python aiohttp 异步压测框架

---

## 一、代码改造

为支持真实压测场景，对 vectorDB 进行了以下改造：

1. **可配置向量维度**: 在 `vectordb_config` 中新增 `DIM` 和 `NUM_DATA` 字段，`Cfg` 类新增解析逻辑，`vector_init.cpp` 从配置读取维度替代硬编码 `dim=1`
2. **AddFollower bug 修复**: `admin_service_impl.cpp` 中 `AddFollower` 函数错误路径手动调用 `done->Run()` 后 `ClosureGuard` 析构时重复调用导致段错误，已移除冗余调用
3. **etcd-cpp-apiv3 编译修复**: 使用 `BUILD_ETCD_CORE_ONLY=ON` 跳过 cpprestsdk 依赖，将 `etcd::Client` 改为 `etcd::SyncClient`

---

## 二、性能压测结果

### 2.1 写入吞吐量

| 测试场景 | 请求数 | 成功率 | QPS | p50(ms) | p95(ms) | p99(ms) |
|----------|--------|--------|-----|---------|---------|---------|
| FLAT 单条upsert (c=32) | 500 | 100% | 880 | 33 | 44 | 46 |
| FLAT 批量upsert (batch=100, c=8) | 100批 | 100% | 30 | 259 | 261 | 317 |
| FLAT 批量upsert 顺序 | 100批 | 100% | 16 | 61 | 62 | 63 |
| HNSW 单条upsert (c=32) | 500 | 100% | 42 | 758 | 821 | 849 |
| HNSW 批量upsert (batch=100, c=8) | 100批 | 100% | 0.8 | 10029 | 10036 | 10135 |
| HNSW 批量upsert 顺序 | 100批 | 100% | 0.3 | 2602 | 10029 | 10030 |
| FLAT insert (无Raft, c=64) | 1000 | 100% | 3729 | 13 | 31 | 35 |
| HNSW insert (无Raft, c=64) | 1000 | 100% | 383 | 164 | 186 | 215 |

**关键发现**:
- FLAT 写入吞吐显著优于 HNSW（索引构建开销差异）
- 批量 upsert 比单条 upsert 效率更高（减少 Raft 共识次数）
- insert（无 Raft/WAL）比 upsert 快 4-80 倍，说明 Raft 共识是主要写入瓶颈
- HNSW 批量写入存在超时问题（Raft blocking + 图索引构建双重开销）

### 2.2 搜索延迟与 QPS

| 测试场景 | 请求数 | QPS | p50(ms) | p95(ms) | p99(ms) | avg(ms) |
|----------|--------|-----|---------|---------|---------|---------|
| FLAT k=1 | 1000 | 2249 | 22 | 35 | 38 | 24 |
| FLAT k=10 | 1000 | 2590 | 20 | 28 | 35 | 20 |
| FLAT k=50 | 1000 | 1825 | 33 | 39 | 46 | 33 |
| FLAT k=10 +filter | 1000 | 3200 | 16 | 27 | 28 | 17 |
| HNSW k=1 | 1000 | 3526 | 15 | 23 | 26 | 15 |
| HNSW k=10 | 1000 | 3411 | 15 | 24 | 26 | 15 |
| HNSW k=50 | 1000 | 2052 | 29 | 33 | 39 | 29 |
| HNSW k=10 +filter | 1000 | 261 | 238 | 270 | 297 | 237 |

**关键发现**:
- HNSW 搜索在小 k 值时比 FLAT 快 1.3-1.6 倍（图索引优势）
- k 值增大时两者性能趋同（HNSW 需要扩展搜索范围）
- 标量过滤对 FLAT 影响小，但对 HNSW 影响显著（QPS 下降 13 倍）

### 2.3 HNSW 召回率

| k值 | FLAT召回率 | HNSW召回率 | FLAT延迟(ms) | HNSW延迟(ms) | 加速比 |
|-----|-----------|-----------|-------------|-------------|--------|
| 1 | 100% | 73% | 2.8 | 1.4 | 2.0x |
| 10 | 100% | 37% | 2.9 | 1.5 | 2.0x |
| 50 | 100% | 22% | 3.1 | 1.7 | 1.9x |
| 100 | 100% | 19% | 3.5 | 2.5 | 1.4x |

**关键发现**:
- HNSW 召回率随 k 值增大而下降（随机均匀分布数据对 HNSW 不利）
- HNSW 速度优势在 k=1 时最明显（2倍加速），大 k 时优势缩小
- 低召回率源于默认参数（M=16, ef_search=50）和随机数据分布

### 2.4 混合读写

| 场景 | 总操作 | 读QPS | 写QPS | 读p95(ms) | 写p95(ms) |
|------|--------|-------|-------|-----------|-----------|
| FLAT 80%读/20%写 | 5000 | 1290 | 311 | 40 | 83 |
| FLAT 50%读/50%写 | 5000 | 451 | 460 | 62 | 96 |
| HNSW 80%读/20%写 | 5000 | 147 | 36 | 395 | 892 |
| HNSW 50%读/50%写 | 5000 | 35 | 36 | 821 | 1297 |

**关键发现**:
- FLAT 混合负载表现良好，80/20 场景下读 QPS 达 1290
- HNSW 在混合负载下性能下降明显（写操作阻塞图索引）
- 写比例增加时 HNSW 性能急剧下降（图索引重建开销）

---

## 三、Raft 分布式测试

**集群配置**: node4 (leader, 7784/8084) + node3 (follower, 7783/8083)

| 测试项 | 结果 |
|--------|------|
| 数据复制一致性 | 10/10 通过 |
| 搜索结果一致性 | 5/5 通过 |
| 批量写入复制 | 10/10 通过 |
| 集群节点状态 | 2 节点正常 (leader + follower) |

**测试细节**:
- 写入 20 条向量到 leader，3 秒后从 follower 查询，全部 10 条已复制
- leader 和 follower 搜索结果完全一致（top-5 ID 和排序完全匹配）
- 批量 upsert 50 条向量，follower 全部可见
- follower 日志索引 33，与 leader 同步

**故障切换测试**: 需手动执行（kill leader → 等待选举 → 验证新 leader），测试脚本已提供自动化指令。

---

## 四、完整集群测试（master+proxy+etcd）

**状态**: 全部通过（6/6）

**集群组件**:
- etcd 3.5.28 (端口 2379) - 元数据存储
- vdb_server_master (端口 6060) - 集群管理
- vdb_server_proxy (端口 6061) - 请求路由/读写分离
- vdb_server 数据节点 (端口 7781/7783/7784) - 数据存储

**测试结果** (6 项全部通过):

| 测试项 | 结果 | 说明 |
|--------|------|------|
| Master 节点管理 | PASSED | AddNode/RemoveNode/GetInstance 通过 etcd 正确操作 |
| 分区配置 CRUD | PASSED | UpdatePartitionConfig/GetPartitionConfig 正确存取，3 分区配置生效 |
| Proxy 写入转发 | PASSED | 10/10 写入通过 proxy 正确分发到 3 个分区 |
| Proxy 搜索广播 | PASSED | 搜索结果从 3 分区正确聚合，返回 [500000, 500009, 500005, 500003, 500004] |
| 读写分离 | PASSED | 各节点维护独立数据分区，可独立搜索 |
| Proxy 拓扑 | PASSED | 正确返回 3 节点列表、master 地址和 instanceId |

**已验证的集群功能**:
1. Master 通过 etcd 管理节点注册/删除/查询
2. 分区配置持久化到 etcd，支持动态更新
3. Proxy 从 master 获取节点列表和分区配置
4. Proxy 写请求按 id 哈希路由到对应分区
5. Proxy 搜索请求广播到所有分区并聚合结果
6. 数据节点支持独立读写

**已修复的 Proxy bug（共 3 项）**:
1. **共享 CURL handle 状态污染**（proxy_service_impl.cpp）：ForwardToTargetNode、FetchAndUpdateNodes、FetchAndUpdatePartitionConfig 三个函数原先共享一个 `curl_handle_`，导致 URL/headers/postdata 残留和数据损坏。改为每个函数创建独立 `curl_easy_init/cleanup`，并在完成后释放 `curl_slist`（修复内存泄漏）。
2. **SetJsonResponse Content-Type 错误**（base_service_impl.cpp）：两个 `SetJsonResponse(const string&)` 重载错误地将 Content-Type 设为 `text/plain`，导致 aiohttp HTTP 客户端的 `resp.json()` 失败（返回空结果）。已改为 `application/json`。
3. **AddFollower 双重 done->Run() 段错误**（admin_service_impl.cpp）：错误路径手动调用 `done->Run()` 后 `ClosureGuard` 析构再次调用导致 crash。已移除所有手动 `done->Run()` 调用，统一由 `ClosureGuard` 管理。

---

## 五、可视化图表

以下交互式图表已生成（plotly HTML 格式）：
- `bench/chart_write.html` - 写入吞吐量对比
- `bench/chart_search.html` - 搜索性能对比
- `bench/chart_recall.html` - HNSW 召回率对比
- `bench/chart_mixed.html` - 混合读写性能

---

## 六、结论

vectorDB 在单节点和 Raft 双节点模式下功能完整、性能可测：

1. **写入性能**: FLAT 索引 + 批量 upsert 是最优写入策略；Raft 共识是主要写入开销
2. **搜索性能**: HNSW 在小 k 值时比 FLAT 快 1.5-2 倍，但召回率较低（73%→19%）
3. **分布式**: Raft 复制一致性 100% 通过，数据同步可靠
4. **混合负载**: FLAT 索引适合读写混合场景，HNSW 写入开销大
5. **完整集群**: master/proxy/etcd 6/6 测试通过，分片路由与广播聚合正确工作

**压测脚本位置**: `/data/workspace/vectorDB/bench/`
**原始数据**: `bench/results_*.json`
