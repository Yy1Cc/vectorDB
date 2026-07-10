# vectorDB 性能压测报告
**生成时间**: 2026-07-09 18:45:16
**向量维度**: 128
**测试环境**: 8核 CPU, TencentOS Server 4.4

## 1. 写入吞吐量测试

| 测试场景 | 请求数 | 成功 | QPS | p50(ms) | p95(ms) | p99(ms) |
|----------|--------|------|-----|---------|---------|---------|
| single_upsert_concurrent [FLAT] n=500 c=32 dim=128 | 500 | 500 | 880.2 | 33.374 | 44.026 | 45.894 |
| batch_upsert_concurrent [FLAT] n=10000 batch=100 c=8 dim=128 | 100 | 100 | 29.9 | 258.966 | 260.522 | 316.515 |
| batch_upsert_sequential [FLAT] n=10000 batch=100 dim=128 | 100 | 100 | 16.3 | 61.076 | 62.021 | 63.152 |
| single_upsert_concurrent [HNSW] n=500 c=32 dim=128 | 500 | 500 | 42.1 | 757.757 | 820.739 | 848.934 |
| batch_upsert_concurrent [HNSW] n=10000 batch=100 c=8 dim=128 | 100 | 100 | 0.8 | 10028.571 | 10036.238 | 10135.085 |
| batch_upsert_sequential [HNSW] n=10000 batch=100 dim=128 | 100 | 100 | 0.3 | 2602.097 | 10028.849 | 10029.706 |
| insert_concurrent [FLAT] n=1000 c=64 dim=128 (no Raft/WAL) | 1000 | 1000 | 3729.0 | 13.265 | 30.645 | 35.144 |
| insert_concurrent [HNSW] n=1000 c=64 dim=128 (no Raft/WAL) | 1000 | 1000 | 382.9 | 163.784 | 186.466 | 215.292 |

## 2. 搜索延迟与 QPS 测试

| 测试场景 | 请求数 | QPS | p50(ms) | p95(ms) | p99(ms) | avg(ms) |
|----------|--------|-----|---------|---------|---------|---------|
| search_concurrent [FLAT] n=1000 k=1 c=64 | 1000 | 2248.7 | 22.119 | 34.797 | 37.652 | 23.516 |
| search_concurrent [FLAT] n=1000 k=10 c=64 | 1000 | 2589.7 | 19.895 | 28.048 | 34.683 | 20.271 |
| search_concurrent [FLAT] n=1000 k=50 c=64 | 1000 | 1825.2 | 32.629 | 38.559 | 45.734 | 33.058 |
| search_concurrent [FLAT] n=1000 k=10 c=64 +filter | 1000 | 3199.5 | 15.878 | 26.69 | 28.06 | 16.739 |
| search_concurrent [HNSW] n=1000 k=1 c=64 | 1000 | 3525.6 | 14.6 | 22.682 | 25.788 | 14.9 |
| search_concurrent [HNSW] n=1000 k=10 c=64 | 1000 | 3411.1 | 15.16 | 23.791 | 26.429 | 15.38 |
| search_concurrent [HNSW] n=1000 k=50 c=64 | 1000 | 2051.9 | 29.207 | 33.101 | 39.171 | 29.432 |
| search_concurrent [HNSW] n=1000 k=10 c=64 +filter | 1000 | 261.1 | 238.463 | 270.232 | 296.627 | 237.132 |

## 3. HNSW 召回率测试

| k值 | FLAT召回率 | HNSW召回率 | FLAT延迟(ms) | HNSW延迟(ms) | 加速比 |
|-----|-----------|-----------|-------------|-------------|--------|
| 1 | 100.00% | 73.00% | 2.784 | 1.371 | 2.03x |
| 10 | 100.00% | 36.86% | 2.901 | 1.45 | 2.0x |
| 50 | 100.00% | 21.88% | 3.123 | 1.688 | 1.85x |
| 100 | 100.00% | 19.14% | 3.476 | 2.528 | 1.37x |

## 4. 混合读写测试

| 测试场景 | 总操作数 | 读QPS | 写QPS | 读p95(ms) | 写p95(ms) |
|----------|----------|-------|-------|-----------|-----------|
| mixed_read_write [FLAT] ops=5000 read=80% c=64 | 5000 | 1290.2 | 310.5 | 40.419 | 83.442 |
| mixed_read_write [FLAT] ops=5000 read=50% c=64 | 5000 | 451.2 | 459.9 | 62.176 | 96.142 |
| mixed_read_write [HNSW] ops=5000 read=80% c=64 | 5000 | 146.9 | 36.0 | 394.754 | 891.782 |
| mixed_read_write [HNSW] ops=5000 read=50% c=64 | 5000 | 35.3 | 35.6 | 820.691 | 1297.356 |

## 可视化图表

- [chart_write.html](./bench/chart_write.html)
- [chart_search.html](./bench/chart_search.html)
- [chart_recall.html](./bench/chart_recall.html)
- [chart_mixed.html](./bench/chart_mixed.html)
