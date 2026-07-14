# vectorDB RocksDB 存储层优化文档

## 概述

本文档记录了 vectorDB 底层 RocksDB 标量存储层的全面优化，涵盖参数调优、存储格式改造和架构分离三个维度，共 7 项可落地优化 + 2 项未实施的学术论文级方案。

**实施时间**：2026 年 7 月

**改动文件**：
- `src/include/database/scalar_storage.h` — 新增 Column Family 句柄、向量操作方法、二进制序列化声明
- `src/database/scalar_storage.cpp` — 完全重写：RocksDB 参数调优、TLV 二进制序列化、CF 分离、向量/标量分离
- `src/database/vector_database.cpp` — 移除死代码、增加 InsertVector/BatchInsertVector 调用、Query 合并 scalar+vector CF

**测试结果**：
- 单节点功能测试：7/7 通过（upsert、query、batch upsert、batch query、search、filter search、update 覆盖）
- 集群测试：6/6 通过（master 管理、分区 CRUD、proxy 转发、proxy 广播、读写分离、拓扑）
- 1000 条批量写入验证：查询准确率 100%，搜索延迟 ~2-3ms

---

## 一、RocksDB 参数调优

### 1.1 Bloom Filter（点查询加速）

```cpp
table_opts.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
```

采用 Full Bloom Filter（10 bits per key），所有 SST 文件共用一个 filter block。点查询时 RocksDB 先查 Bloom Filter，判断 key 是否可能存在，避免无谓的磁盘 IO。10 bits/key 在 128 维向量 + 标量字段的场景下，假阳性率约为 1%。

### 1.2 压缩（Per-Level LZ4）

```cpp
opts.compression_per_level = {
    rocksdb::kNoCompression, rocksdb::kNoCompression, rocksdb::kNoCompression,
    rocksdb::kLZ4Compression, rocksdb::kLZ4Compression, rocksdb::kLZ4Compression,
    rocksdb::kLZ4Compression
};
```

L0-L2 不压缩，保证写入路径的低延迟；L3-L6 使用 LZ4 压缩，在高层的冷数据上节省磁盘空间。LZ4 是 RocksDB 编译时唯一可用的压缩库（无 ZSTD/Snappy），其特点是压缩比中等但速度极快（~500MB/s 解压）。

### 1.3 后台线程

```cpp
opts.max_background_jobs       = 12;
opts.max_background_flushes    = 4;
opts.max_background_compactions = 8;
```

RocksDB 8.0 使用 `max_background_jobs` 统一管理后台线程池，自动在 flush 和 compaction 之间分配。设置 12 个后台线程、最多 4 个并发 flush、8 个并发 compaction，充分利用多核 CPU。

### 1.4 Block Cache（512MB LRU）

```cpp
table_opts.block_cache = rocksdb::NewLRUCache(512ULL * 1024 * 1024);
table_opts.cache_index_and_filter_blocks = true;
table_opts.pin_l0_filter_and_index_blocks_in_cache = true;
table_opts.block_size = 16 * 1024;  // 16KB data blocks
```

分配 512MB LRU Block Cache 缓存热点数据块。`cache_index_and_filter_blocks` 将 index 和 filter blocks 也纳入 block cache 管理（而非独立的内存），`pin_l0` 确保 L0 的 index/filter 不会被淘汰（L0 查询频率最高）。Block size 设为 16KB（默认 4KB），匹配 128 维向量 + 标量的典型 value 大小。

### 1.5 BlobDB（KV 分离）

```cpp
opts.enable_blob_files     = true;
opts.min_blob_size         = 512;
opts.blob_compression_type = rocksdb::kLZ4Compression;
opts.enable_blob_garbage_collection = true;
```

启用 RocksDB 的 BlobDB 特性（KV 分离）：大于 512 字节的 value 写入独立的 blob 文件，SST 文件只存储 key + blob 引用指针。Compaction 时不需要重写 blob 数据，显著减少写放大。Blob 文件使用 LZ4 压缩，开启 GC 自动清理过期 blob。

### 1.6 Dynamic Level Bytes

```cpp
opts.level_compaction_dynamic_level_bytes = true;
```

开启后 RocksDB 根据实际数据量动态调整各层大小，将约 90% 的数据下沉到最后一层，减少多层之间的数据重叠，降低 compaction 频率和写放大。

### 1.7 Write Buffer 256MB

```cpp
opts.write_buffer_size             = 256ULL * 1024 * 1024;
opts.max_write_buffer_number       = 4;
opts.min_write_buffer_number_to_merge = 2;
```

单 MemTable 大小设为 256MB，最多 4 个活跃 MemTable（含 1 个 active + 3 个 immutable），flush 时至少合并 2 个 immutable MemTable。大 MemTable 减少 flush 频率，适合批量写入场景。

### 1.8 WAL 512MB

```cpp
opts.max_total_wal_size = 512ULL * 1024 * 1024;
```

WAL 总大小上限 512MB，确保写入持久性的同时避免 WAL 文件无限增长。

---

## 二、存储格式改造

### 2.1 二进制 TLV 格式替代 JSON

旧方案将完整 JSON 文档（含向量数组序列化为文本）直接写入 RocksDB。新方案使用自定义 TLV 二进制格式：

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| Magic  |Version |NumFields|  Field1 TLV                           |
| (1B)   | (1B)   | (2B)   | namelen(2B)+name+type(1B)+vallen(4B)+val|
+--------+--------+--------+--------+--------+--------+--------+--------+
```

字段类型支持：
- `kTypeInt64 (0)` — int64
- `kTypeUint64 (1)` — uint64
- `kTypeString (2)` — string
- `kTypeDouble (3)` — double
- `kTypeBool (4)` — bool

序列化时自动排除 `vectors` 字段（向量单独存储），只序列化标量字段。所有整数采用 little-endian 编码，无需外部序列化依赖。

### 2.2 向量原始 float32 存储

向量不再作为 JSON 数组文本存储，改为直接写入原始 float32 二进制数据：

```cpp
static auto SerializeVector(const float* data, size_t count) -> std::string {
    return std::string(reinterpret_cast<const char*>(data), count * sizeof(float));
}
```

128 维向量固定为 512 字节，无需解析。

### 2.3 存储量对比

| 方案 | 标量存储 | 向量存储 | 合计（128维） |
|------|----------|----------|---------------|
| 旧方案（JSON） | ~2000B（含向量 JSON） | 包含在 JSON 中 | ~2000B |
| 新方案（TLV+float32） | ~40B（TLV 标量字段） | 512B（raw float32） | ~552B |

存储量减少约 72%。

---

## 三、架构分离

### 3.1 Column Family 分离

RocksDB 实例维护三个 Column Family，各自独立调优：

| CF 名称 | 用途 | Key 格式 |
|---------|------|----------|
| `default` | 系统预留（未使用） | — |
| `scalar` | 标量字段（TLV 二进制） | `collection:id` → TLV bytes |
| `vector` | 向量数据（raw float32） | `collection:id` → float32 bytes |

`create_missing_column_families = true` 确保首次运行时自动创建。

### 3.2 向量/标量分离

旧方案中向量和标量共存在同一个 JSON value 中，每次查询都需要反序列化完整的 JSON（含 128 个浮点数文本）。

新方案分离后的数据流：

**写入（Upsert）**：
1. `InsertScalar` → 将标量字段（排除 vectors）序列化为 TLV，写入 scalar CF
2. `InsertVector` → 将向量字段序列化为 raw float32，写入 vector CF

**查询（Query）**：
1. `GetScalar` → 从 scalar CF 读取 TLV 二进制，反序列化为 JSON Document
2. `GetVector` → 从 vector CF 读取 float32 二进制，反序列化为 float 数组
3. 合并：将 vectors 数组注入 JSON Document，返回完整文档

### 3.3 移除死代码

在 `Upsert` (vector_database.cpp:164-169) 中，原代码从 ScalarStorage 读取旧向量的 float 数组，但实际上 `RemoveVectors` 只使用 ID 进行索引删除，旧向量数据从未被使用。此段死代码已移除。

---

## 四、未实施的方案

以下两个方案属于学术论文级别的内核改造，无法通过 RocksDB 配置参数直接应用，暂未实施：

### 4.1 ArceKV（动态 Compaction）

来源：NTU Siqiang Luo 组，PVLDB Vol.19 (2026)

核心思路：传统 RocksDB Leveled Compaction 为静态工作负载设计（L0 tiering + L1+ leveling），读写比变化时无法自适应切换。ArceKV 提出 ElasticLSM 去掉层的结构约束 + Arce 决策引擎实时选择最优 compaction 策略，在动态负载下性能提升约 3 倍。

实施难度：需要深入修改 RocksDB 内核 10-20 个核心文件（CompactionPicker、Version、VersionEdit、DBImpl 等），新增三种 compaction pattern（Intra-level、Adjacent-level、Multi-level），实现 Windowed-State Cost Model 和候选剪枝算法。预估工作量 2-4 周，且需大量回归测试。

### 4.2 Resystance（eBPF + io_uring）

来源：iWiki 文档

核心思路：利用 eBPF 将 compaction 逻辑下沉到内核态，配合 io_uring 实现零拷贝异步 IO。属于内核级改造，不适合作为参数配置项。

---

## 五、验证方法

### 5.1 确认所有参数生效

启动服务后查看 RocksDB OPTIONS 文件：

```bash
cat /root/vectordb1/storage/OPTIONS-* | grep -E "write_buffer_size|compression|enable_blob|bloom|block_cache|max_background"
```

### 5.2 确认 Column Family 创建

```bash
cat /root/vectordb1/storage/OPTIONS-* | grep "column_family\|name="
```

应看到 default、scalar、vector 三个 CF。

### 5.3 功能验证

```bash
python3 bench/cluster_test.py   # 集群测试 6 项
# 单节点 API 测试：upsert → query → search → filter search → update
```
