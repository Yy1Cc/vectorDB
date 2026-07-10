# VectorDB 功能增强更新

本次更新在原有分布式向量数据库基础上，实现了向量量化、内积距离、增强过滤、分层存储、全文检索、TTL 过期管理等多项核心功能，并接入 HTTP API 主流程供用户直接使用。共新增 7 个测试套件（54 个测试用例），全量编译通过。

## 功能清单

### 1. SQ8/SQ4 标量量化索引
基于 faiss `IndexScalarQuantizer` 实现 SQ8（4x 压缩）和 SQ4（8x 压缩）量化索引。采用 lazy train 机制，首次插入时自动训练量化器。通过 `indexType: "SQ8"` / `"SQ4"` 选择，显著降低内存占用。

### 2. 内积距离 + ip2cos 预处理
新增 `InnerProductSpace` 模块，提供 IP_FLAT / IP_SQ8 索引类型。采用 ip2cos 方案：对向量做 L2 归一化后，内积等价于余弦相似度，复用 faiss 的 L2/内积搜索路径，保证距离一致性。

### 3. 增强标量过滤
`FilterIndex` 扩展支持 6 种比较操作（=、!=、>、<、>=、<=）和字符串字段过滤。搜索接口支持 `filters` 数组实现多条件 AND 组合，兼容原有单条件 `filter` 参数。

### 4. 分层存储
新增 `LayeredIndex`，由内存流式部分（StreamingPart，暴力搜索）和基础 faiss 索引组成。写入先进流式部分，达到阈值自动 flush 到基础索引，兼顾写入吞吐与搜索性能。提供 LAYERED_FLAT / LAYERED_SQ8 两种类型。

### 5. BM25 全文检索
新增 `FullTextIndex`，基于倒排索引实现 BM25 评分（k1=1.2, b=0.75）。Upsert 时自动对字符串字段建索引；搜索时通过 `fulltext` 参数指定查询词，结果转为 bitmap 与向量搜索过滤条件做 AND 交集。

### 6. TTL 过期管理
新增 `TTLManager`，支持按向量设置 TTL 或绝对过期时间戳。采用惰性过滤策略：搜索结果中标记过期 ID，不阻塞搜索主路径。可通过 `CleanExpiredVectors` 主动清理。

### 7. HTTP API 集成
上述功能全部接入 HTTP API 主流程：
- Upsert / BatchUpsert 自动识别 `indexType`、字符串字段、`ttl` 字段
- Search 支持 `indexType`、`filters`（多条件）、`fulltext`、`fieldType` 参数
- 所有新功能向后兼容，默认使用 FLAT + L2

## 压测与测试
- `bench/` 目录包含 QPS、召回率、混合负载、Raft 一致性压测脚本
- `STRESS_TEST_REPORT.md` / `BENCHMARK_REPORT.md` 记录压测结果
- 7 个测试套件覆盖量化、内积、过滤、分层存储、全文+TTL、集成、端到端，共 54 个用例全部通过

## 涉及文件
- 新增：`inner_product_space`、`layered_index`、`fulltext_index`、`ttl_manager`、`etcd_http_client`
- 修改：`index_factory`、`faiss_index`、`filter_index`、`vector_database`、各 HTTP service、`constants`、`vector_init`、CMake 配置
