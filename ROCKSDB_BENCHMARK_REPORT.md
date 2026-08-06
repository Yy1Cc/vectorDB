# RocksDB 存储层对比压测报告

测试日期：2026-08-05
测试程序：`test/index/rocksdb_storage_bench_test.cpp`
环境：vectorDB 自带 RocksDB（LZ4 only，无 ZSTD/Snappy）

## 1. TLV 二进制序列化 vs JSON 序列化（N=10000）

典型标量文档：id + int_field + label(字符串) + price(int64) + active(bool) + score(double)。

| 指标            | TLV        | JSON       | TLV/JSON |
|-----------------|------------|------------|----------|
| 平均 size (B)   | 116.5      | 86.6       | 1.35x    |
| 总 size (KB)    | 1137.7     | 845.2      | 1.35x    |
| 序列化 ops/ms   | 228.2      | 381.1      | 0.60x    |
| 反序列化 ops/ms | 611.5      | 304.7      | 2.01x    |

**反直觉发现**：TLV 比 JSON **大** 1.35x，不是更小。原因：标量字段多为小整数，JSON 文本表示紧凑（`123` 占 3 字节），而 TLV 每字段固定 8 字节 value + 1 字节 type + 4 字节 len + 2 字节 name_len = 15 字节固定 overhead，对小值反而更占空间。TLV 的 size 优势只在大 value（长字符串、大数组）时才成立。

TLV 的真实价值在**反序列化速度**（2.01x）和**类型保留**（JSON 会丢失 int/double 精度区分，TLV 保留类型码），不在省空间。序列化速度 TLV 慢 0.60x，因为要逐字段判类型+拼二进制，rapidjson Writer 的流式输出反而更快。

## 2. BlobDB on/off 对比（N=20000, 128维 float32 = 512B/value）

user payload = 10.07 MB。

| 配置        | 写入(MB) | QPS        | 写放大  | 空间放大 | 耗时(s) |
|-------------|----------|------------|---------|----------|---------|
| BlobDB off  | 0*       | 133385.9   | 0*      | 1.01     | 0.15    |
| BlobDB on   | 0*       | 107184.1   | 0*      | 1.10     | 0.19    |

*写放大指标 `rocksdb.bytes-written` 在本 RocksDB 版本下 GetIntProperty 返回 0（property 不支持），改用空间放大（live-data-size/user_payload）作为替代指标。

**反直觉发现**：BlobDB on 反而**更慢 20%**（133386→107184 QPS）、**空间更大 9%**（1.01→1.10）。原因：小数据集（10MB）+ 单轮 flush，没有多轮 compaction 产生写放大，BlobDB 的收益（减少大 value 的 compaction 重写）无从发挥，只有开销（blob 文件索引、分离写路径、blob GC）。这与 SQ8 在小数据集下比 FLAT 慢是同一类问题——**收益是规模依赖的，小数据集测不出**。BlobDB 真正见效需要百万级数据 + 多轮 compaction。

## 3. 压缩 LZ4 vs None 空间对比（N=20000）

user payload = 10.07 MB。

| 配置             | 磁盘占用(MB) | 空间比 |
|------------------|--------------|--------|
| NoCompression    | 10.15        | 1.01   |
| LZ4 (per-level)  | 10.15        | 1.01   |

**反直觉发现**：LZ4 压缩对向量数据**完全无效**（压缩比 1.0）。原因：向量是 raw float32，浮点数据随机性高，LZ4 找不到重复模式，压不动。LZ4 只对标量字段（TLV 二进制/JSON 文本）有压缩效果，但标量占比远小于向量，整体压缩比≈1。

## 诚实结论：RocksDB 调优在当前向量场景的收益有限

三组测试数据都不支持"RocksDB 调优显著提升性能"的叙事：
- TLV：反序列化快 2x（真实优势），但空间更大 1.35x，序列化更慢 0.60x
- BlobDB：小数据集负收益（慢 20%、空间大 9%），收益要大数据集才显现
- LZ4：对向量浮点无效（压缩比 1.0）

这些发现与 SQ8 在小数据集下比 FLAT 慢是同一根因：**优化收益是规模/数据类型依赖的**。在 1-2 万条小数据集 + 浮点向量场景下，压缩、KV 分离、量化的收益都无法兑现，甚至有净开销。

## 面试讲法（诚实版）

不要说"我的 RocksDB 调优提升了性能"——数据不支持。应说：

> 我对存储层做了系统压测验证。测下来发现，在向量场景下这些调优的收益有限甚至为负：LZ4 对浮点向量压缩比是 1.0（压不动），BlobDB 在小数据集下因为没多轮 compaction 反而慢 20%，TLV 序列化比 JSON 大 1.35x 但反序列化快 2x。这些反直觉结果说明优化收益是规模和数据类型依赖的——量化的带宽优势、BlobDB 的写放大优势都要百万级数据才显现。这个实测过程本身比"我开了什么配置"更有价值，因为它让我理解了每个优化的生效边界。

这体现的是"实测验证而非想当然"的工程素养，比声称"调优有效"但拿不出数据更有说服力。面试官追问"那你觉得什么场景下这些优化才有效"，你能答出"百万级数据 + 多轮 compaction 后 BlobDB 才减少写放大、浮点不可压所以压缩对向量无效该用量化而非压缩"——这是有数据支撑的深度。

## 已修复的 bug

`scalar_storage.cpp:127` 日志写 "ZSTD compression" 但实际配置是 LZ4（RocksDB 构建时只有 LZ4），已修正为 "LZ4 compression"。

## 4. 大数据集 BlobDB on/off 对比（1M 条，验证收益随规模反转）

写放大指标修正：`BYTES_WRITTEN` 是 Statistics 的 ticker（非 GetIntProperty property，之前用错 API 已修正），用 `statistics->getTickerCount(BYTES_WRITTEN)` 读取。`big-ann-benchmarks` 仓库无数据文件（只有准备脚本），用生成的 1M 条 128 维向量（503.5MB）+ 16MB memtable 测试。

| 配置        | 写入(MB) | QPS        | 写放大 | 空间放大 | 耗时(s) |
|-------------|----------|------------|--------|----------|---------|
| BlobDB off  | 1297.1   | 133909     | 2.58   | 0.47     | 7.5     |
| BlobDB on   | 785.6    | 181126     | 1.56   | 1.09     | 5.5     |

**关键发现：BlobDB 写放大降低 40%，QPS 快 35%**。持续写入 1M 条（512MB）+ 关闭 dynamic level + 小 memtable(16MB) 触发多级 compaction，用 `FLUSH_WRITE_BYTES + COMPACT_WRITE_BYTES` ticker 测真实写放大。BlobDB off 写放大 2.58（LSM 含大 value，compaction 反复重写大 value），on 写放大 1.56（LSM 只含小索引，compaction 只重写小索引，大 value 在 blob 不参与 compaction）。差值 1297-785=512MB ≈ 一轮 payload，正好是一轮大 value 的 compaction 重写被 BlobDB 省掉。QPS on 快 35%（181126 vs 133909），因 compaction 重写量小。

**指标修正历程**：最初用 `GetIntProperty("rocksdb.bytes-written")` 返回 0（property 名不存在）→ 改用 `BYTES_WRITTEN` ticker 写放大恒 1.00（该 ticker 只统计 user Put 不含 compaction）→ 最终用 `FLUSH_WRITE_BYTES + COMPACT_WRITE_BYTES` 才测出真实 compaction 写放大。`WaitForCompact` API 该版本不存在，改用轮询 `compaction-pending` property 等后台 compaction 完成。

**收益随规模对照**：20K 条（10MB）BlobDB on 慢 21%、写放大无差异（无多轮 compaction 可减）；1M 条（512MB）on 写放大降 40%、QPS 快 35%。量化了 BlobDB 收益是规模依赖的。
