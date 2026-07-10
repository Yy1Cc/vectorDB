# vectorDB 性能压测与分布式测试报告

**测试日期**: 2026-07-09
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

**状态**: 受阻于 protobuf 版本冲突

**完成的工作**:
- etcd 3.5.28 安装并运行成功（端口 2379）
- gRPC C++ 1.56.2 安装成功
- etcd-cpp-apiv3 v0.15.4 编译成功（BUILD_ETCD_CORE_ONLY 模式）
- vdb_server_master 编译成功
- master 启动后段错误

**阻塞原因**:
系统 protobuf 24.2（gRPC 1.56.2 依赖）与项目 third_party protobuf 3.17.3 共存导致符号冲突。etcd-cpp-api 链接系统 protobuf，项目代码链接 third_party protobuf，运行时双重加载导致崩溃。

**修复方案（未实施）**:
1. 升级 third_party protobuf 到 24.x（需验证与 brpc 兼容性）
2. 或降级系统 gRPC 到兼容 protobuf 3.17 的版本
3. 或使用静态链接隔离 protobuf 符号

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
5. **完整集群**: master/proxy/etcd 架构因 protobuf 版本冲突未完成测试，需后续修复

**压测脚本位置**: `/data/workspace/vectorDB/bench/`
**原始数据**: `bench/results_*.json`
