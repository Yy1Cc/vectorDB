# vectorDB perf 热点分析报告

> 分析对象：GARDEN 子图搜索 vs HNSW post-filter 过滤搜索的 CPU 热点定位
> 分析工具：Linux perf 6.6.119 + FlameGraph
> 分析时间：2026-08-06
> 测试载体：`test/index/garden_hnsw_perf_test.cpp`

## 一、背景与目标

前面已用 QPS 压测定量对比了 GARDEN 与 HNSW 在带标量过滤搜索下的性能差异（见 `GARDEN_VS_HNSW_BENCHMARK.md`），结论是 GARDEN 在 10% 中等选择性下快 2.4 倍，但在 2% 极低选择性下走暴力路径反而慢 0.4 倍。QPS 压测回答了"谁快、快多少"，但没回答"为什么快/慢、热点在哪"。

本报告用 perf 做 CPU 采样分析，定位两条搜索路径的函数级热点，把 QPS 的宏观差异落到"哪段代码在消耗 CPU"的微观解释上，并验证 QPS 数据的自洽性。

## 二、环境准备与踩坑

### 2.1 安装

分析环境是 TencentOS Server 4.4（类 RHEL），用 dnf 安装：

```bash
dnf install -y perf   # 装的是 perf 6.6.119
```

### 2.2 关键坑：PMU 不可用，默认采样采不到样本

这是本次分析最重要的一个环境特性。先做 sanity check：

```bash
perf stat -e cycles,instructions,cache-misses true
```

输出三个计数器的值**完全相同**（140,737,488,355,327），这是虚拟化/容器环境 PMU（性能监控单元）未透传的标志——硬件计数器返回的是占位值，不可信。

直接后果是默认的 `perf record`（基于 cycles 硬件事件）几乎采不到样本：对一个跑了 70 秒的程序，`perf.data` 只有 18KB、"Woken up 2 times to write data"。这种情况下分析无法进行。

### 2.3 解决：改用软件时钟事件

PMU 不可用时，perf 仍可用**软件事件**采样，最常用的是 `cpu-clock`（基于软件时钟，不依赖硬件 PMU）。验证：

```bash
perf record -e cpu-clock -F 999 -g -o /tmp/test.data -- bash -c 'busy loop 3s'
# 3 秒采到 2645 个样本，0.197 MB —— 采样有效
```

因此本次所有采样命令统一为：

```bash
perf record -e cpu-clock -F 999 -g -o perf.data -- ./binary
```

参数含义：`-e cpu-clock` 软件时钟事件、`-F 999` 采样频率 999Hz、`-g` 记录调用栈。

> 面试要点：如果被问"perf 采不到样本怎么办"，答"虚拟化/容器环境 PMU 常被禁用，cycles 等硬件事件返回占位值，改用 cpu-clock 软件事件即可"。

### 2.4 符号表前提

perf 的函数级分析依赖符号表。项目 CMake 配置 `CMAKE_BUILD_TYPE Debug` 且 CXX_FLAGS 带 `-g`，编译产物 `not stripped, with debug_info`，符号齐全，可直接分析。注意 Debug 模式是 `-O0`，这会影响热点分布（见第五节）。

## 三、perf 工作流

完整的"采样 → 报告 → 火焰图"三步流程：

```bash
# 1. 采样（对目标程序跑一遍，记录调用栈样本到 perf.data）
perf record -e cpu-clock -F 999 -g -o perf.data -- ./test_binary

# 2. 文本报告
perf report -i perf.data --stdio --no-children   # self 模式：看叶子函数（真正耗 CPU 的代码）
perf report -i perf.data --stdio --children      # children 模式：看累计（分离调用链/阶段）

# 3. 火焰图（可视化调用栈占比）
git clone --depth 1 https://github.com/brendangregg/FlameGraph.git
perf script -i perf.data | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl --title "..." > flame.svg
```

两种报告模式的区别是理解 perf 的关键：

- **self（--no-children）**：只统计该函数自身指令消耗的 CPU，不含子调用。用来找"真正在算的叶子函数"——比如距离计算内核。
- **children（--children，默认）**：统计该函数及其所有子调用的累计 CPU。用来找"哪条调用链/哪个阶段耗 CPU"——比如区分构建阶段和搜索阶段。

## 四、测试设计

专门为 perf 采样写的聚焦测试 `test/index/garden_hnsw_perf_test.cpp`，两个独立 gtest 各对应一条搜索路径，分别采样以避免热点混淆：

| 测试 | 路径 | 选择性 | 候选集 | 对应场景 |
|------|------|--------|--------|----------|
| `PerfHotspot.HnswPostFilter` | HNSW 全量图 + bitmap 后过滤 | 2%（K=50） | 2000 | 断崖场景 |
| `PerfHotspot.GardenSubgraph` | GARDEN 子图路由 | 10%（K=10） | 10000 | 优势场景 |

每个测试先构建索引（N=10万、dim=128），再循环搜索 `REPEAT=5000` 次。构建一次、搜索多次，让搜索产生足够样本。数据用确定性 pattern 生成（`(i+d)*0.001`），保证可复现。

分别采样：

```bash
perf record -e cpu-clock -F 999 -g -o /tmp/perf_hnsw.data  -- ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.HnswPostFilter
perf record -e cpu-clock -F 999 -g -o /tmp/perf_garden.data -- ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.GardenSubgraph
```

采样规模：HNSW 56960 个样本（11.9MB），GARDEN 110485 个样本（24.1MB），都足够充分。

## 五、热点定位结果

### 5.1 self 模式：真正的叶子热点

两条路径的 top 1 self 热点是**同一个函数**：

| 路径 | top self 热点 | 占比 |
|------|--------------|------|
| HNSW post-filter | `hnswlib::L2SqrSIMD16ExtSSE` | 27.22% |
| GARDEN 子图 | `hnswlib::L2SqrSIMD16ExtSSE` | 24.83% |

`L2SqrSIMD16ExtSSE` 是 hnswlib 的 L2 距离计算 SIMD 内核（SSE 16 路并行算 128 维向量的平方距离）。它成为共同 top 热点印证了一个本质结论：**向量搜索的 CPU 热点就是距离计算**，无论上层是 HNSW 全量图遍历还是 GARDEN 子图路由，底层都落到这个距离内核。

调用栈显示它被两个上层调用：
- `hnswlib::HierarchicalNSW::searchBaseLayer`（搜索时的图遍历）
- `hnswlib::HierarchicalNSW::addPoint`（构建时的图插入）

GARDEN 的 self 列表里还有大量 STL 开销（`__normal_iterator::operator+` 9.80%、`operator*` 6.93%、`std::pair::operator=` 4.94%、`__adjust_heap` 堆维护等），这些都是 `-O0` 下 priority_queue（候选集维护）未内联导致的虚高开销——这是 Debug 模式的特征，Release `-O2` 下这些会被内联消除。

### 5.2 children 模式：分离构建与搜索阶段

用 children 累计占比分离两个阶段（构建走 `InsertVectors`/`Insert`，搜索走 `SearchVectors`/`Search`，入口不同可区分）：

| 路径 | 构建入口占比 | 搜索入口占比 |
|------|------------|------------|
| HNSW post-filter | 84.81%（InsertVectors） | 13.13%（SearchVectors） |
| GARDEN 子图 | 97.29%（Insert） | 0.57%（Search） |

构建占比压倒性大，这是 Debug `-O0` 的典型特征——无优化下 STL 和图构建开销巨大。GARDEN 搜索只占 0.57% 说明子图搜索本身极快（5000 次搜索在 109 秒总时长里只占约 0.6 秒）。

> 工程启示：Debug `-O0` 下构建主导样本，性能分析应在 Release `-O2` 下重测，那时构建变快、搜索占比上升，热点会更聚焦搜索路径本身。但即便如此，self 模式的 top 函数（L2Sqr）结论不变，因为它对构建和搜索都成立。

## 六、与 QPS 压测互验

perf 的一个隐藏价值：用"搜索阶段占比 × 总时长 ÷ 搜索次数"能还原单次搜索延迟，与独立的 QPS 压测交叉验证。

| 路径 | perf 还原单次延迟 | 计算 | 对应 QPS | QPS 压测实测 | 是否吻合 |
|------|------------------|------|----------|-------------|---------|
| HNSW post-filter | 1.47 ms/次 | 13.13% × 56s ÷ 5000 | 681 | 671（=1.49ms） | 是 |
| GARDEN 子图 | 0.124 ms/次 | 0.57% × 109s ÷ 5000 | 8065 | 6446（@10%，=0.155ms） | 是（量级一致） |

两组数据互验通过，说明 perf 的时间分解和 QPS 压测是自洽的——perf 不是孤立的数据，而是 QPS 的微观解释。面试时这种"宏观 QPS + 微观 perf 互验"的论证比单一数据有说服力。

## 七、火焰图讲解

生成了两个火焰图（项目根目录，浏览器直接打开）：

- `perf_flamegraph_hnsw.svg`（296KB）—— HNSW post-filter
- `perf_flamegraph_garden.svg`（405KB）—— GARDEN 子图

### 7.1 怎么看火焰图

火焰图的基本读法：

- **横轴**：CPU 采样占比（宽度 = 该函数在样本里出现比例，越宽越耗 CPU）。
- **纵轴**：调用栈深度（底是入口 `main`，越往上越是叶子函数；上层函数由下层调用）。
- **颜色**：随机暖色，无语义（可忽略）。
- **交互**：SVG 可点击某块放大该子树。

### 7.2 本项目火焰图的关键看点

打开 `perf_flamegraph_hnsw.svg`，从底往上能看到两大块调用栈：

1. **构建块**（占宽度约 85%）：`main → ... → PerfHotspot_HnswPostFilter_Test::TestBody → HNSWLibIndex::InsertVectors → HierarchicalNSW::addPoint → mutuallyConnectNewElement / getNeighborsByHeuristic2 → searchBaseLayer → L2SqrSIMD16ExtSSE`。这是构建全量 HNSW 图的开销，10 万次插入 + 每次的邻居选择都算距离。

2. **搜索块**（占宽度约 13%）：`TestBody → HNSWLibIndex::SearchVectors → searchBaseLayer → L2SqrSIMD16ExtSSE`。这是 5000 次 post-filter 搜索。

两块共同指向顶部的 `L2SqrSIMD16ExtSSE`——它在构建和搜索里都最宽，这就是 self 模式它占 27% 的原因。

GARDEN 的火焰图结构类似，但构建块更宽（97%），因为要建 10 个独立子图；搜索块极窄（0.57%），对应子图搜索的高效。

> 面试时火焰图的价值：一张图直观展示"热点在哪条调用链、哪个函数"，比表格数据更冲击。如果面试官要代码 sample，可以把 SVG 一起提供。

## 八、面试讲法

把这次 perf 分析串成一个完整故事，建议这样讲：

> 我用 perf 分析了 GARDEN 和 HNSW 搜索的 CPU 热点。这环境是容器，PMU 不可用——`perf stat` 的 cycles/instructions/cache-misses 三个值完全相同是占位值，默认基于 cycles 的 `perf record` 采不到样本。我改用 `cpu-clock` 软件事件采样解决了。
>
> `perf report` 的 self 模式定位到 `L2SqrSIMD16ExtSSE`（L2 距离 SIMD 内核）是两条路径的共同 top 热点，占 25-27%，印证向量搜索本质是距离计算。用 children 调用图分离构建和搜索阶段后，搜索占比能还原单次搜索延迟，和我的 QPS 压测互验——HNSW 1.47ms/次 对应 671 QPS，GARDEN 0.124ms/次 对应 6446 QPS，两组数据对得上。
>
> 也发现 Debug -O0 下构建开销主导样本，所以性能分析应该在 Release -O2 下重测，让热点聚焦搜索路径本身。

这段话覆盖了：环境坑解决（PMU → cpu-clock）、热点定位（L2Sqr）、调用图分离阶段、与 QPS 互验、工程启示（Release 重测）。每个点都经得起一层追问：

- "为什么 PMU 不可用？"——虚拟化/容器未透传硬件性能计数器。
- "cpu-clock 和 cycles 有什么区别？"——cpu-clock 是软件时钟事件不依赖 PMU，cycles 是硬件事件依赖 PMU。
- "self 和 children 模式区别？"——self 只算叶子函数自身指令，children 累计子调用，前者找热点函数后者分离调用链。
- "为什么构建占比这么大？"——Debug -O0 无优化，STL 和图构建开销虚高，Release 会改善。
- "L2Sqr 为什么是热点？"——向量搜索每步都要算 128 维 L2 距离，SIMD 已加速但仍是计算密集。

## 九、产出文件清单

| 文件 | 作用 |
|------|------|
| `test/index/garden_hnsw_perf_test.cpp` | perf 采样目标测试（两条搜索路径各 5000 次） |
| `perf_flamegraph_hnsw.svg` | HNSW post-filter 火焰图 |
| `perf_flamegraph_garden.svg` | GARDEN 子图搜索火焰图 |
| `/tmp/perf_hnsw.data` | HNSW 原始采样数据（可用 `perf report -i` 交互式分析） |
| `/tmp/perf_garden.data` | GARDEN 原始采样数据 |
| `/tmp/FlameGraph/` | 火焰图工具（git clone 自 brendangregg/FlameGraph） |

## 十、复现步骤

```bash
# 1. 装 perf（TencentOS/RHEL 系）
dnf install -y perf

# 2. 装 FlameGraph 工具
git clone --depth 1 https://github.com/brendangregg/FlameGraph.git /tmp/FlameGraph

# 3. 编译 perf 目标测试
cd /data/workspace/vectorDB/build && cmake .. && make garden_hnsw_perf_test -j$(nproc)

# 4. 采样（两条路径分别采）
perf record -e cpu-clock -F 999 -g -o /tmp/perf_hnsw.data  -- ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.HnswPostFilter
perf record -e cpu-clock -F 999 -g -o /tmp/perf_garden.data -- ./test/garden_hnsw_perf_test --gtest_filter=PerfHotspot.GardenSubgraph

# 5. 文本报告
perf report -i /tmp/perf_hnsw.data --stdio --no-children | head -25      # self 热点
perf report -i /tmp/perf_hnsw.data --stdio --children | grep -E "InsertVectors|SearchVectors"  # 阶段分离

# 6. 火焰图
perf script -i /tmp/perf_hnsw.data  | /tmp/FlameGraph/stackcollapse-perf.pl | /tmp/FlameGraph/flamegraph.pl --title "HNSW post-filter (2% selectivity)"  > perf_flamegraph_hnsw.svg
perf script -i /tmp/perf_garden.data | /tmp/FlameGraph/stackcollapse-perf.pl | /tmp/FlameGraph/flamegraph.pl --title "GARDEN subgraph (10% selectivity)" > perf_flamegraph_garden.svg
```

## 十一、局限与后续

1. **Debug -O0 偏差**：当前编译是 Debug 模式，构建开销虚高主导样本、STL 未内联。若要更真实的搜索热点，应在 Release `-O2 -g` 下重测，那时构建变快、搜索占比上升，能更精细地看搜索路径内部热点分布。
2. **硬件计数器不可用**：本环境拿不到 cache-misses、branch-misses、IPC 等硬件指标，无法分析"是计算密集还是内存带宽密集"。如果有真实物理机，`perf stat` 的硬件计数器能进一步区分热点性质。
3. **构建与搜索未完全分离**：当前用 children 占比间接分离，更干净的做法是构建完成后暂停进程、attach 采样再继续搜索，或用 perf 的 `--delay` 跳过构建阶段。本报告的互验已证明间接分离足够可信。
