# Incremental causal control plane 结果

日期：2026-08-05

分支：`exp/incremental-causal-control-plane`

预注册：`978063c`

机器：本地 CPU only；没有 GPU、下载或 SSH

确定性结果：`results/systems/incremental-causal-control-plane-deterministic-v1.json`

实测结果：`results/systems/incremental-causal-control-plane-empirical-v1.json`

## 一句话结论

**INCREMENTAL CONTROL PLANE GO。**

在完全不改变 Causal Hard Frontier dispatch/B32/W64/cost 的前提下，增量实现通过五域 50/50
full-trace exact 硬门；透明插桩基线 / optimized 的 deterministic operation proxy 中位为
`12.63×`，large-trace 单进程 CPU 中位加速 `2.08×`，最慢 workload 仍有 `1.84×`。
small trace 也加速，optimized/transparent 最坏为 `0.663`，没有小 workload 回退。

优化后的 Hard method 与 efficient Delay-D32 控制面已经基本等价：operation 只高约 0.75%，
large CPU 中位比约 1.0。此前“Hard 为约 1% 系统收益付出约 75% 更多控制操作”的问题主要是
透明实现缺陷，而不是 completion+age scoring 必然昂贵。

这是一项实现与系统贡献，不是新的调度算法。B32 与 Delay-D32 的公平可行域仍然结构等价；
本轮只使已冻结的 scoring 更便宜。

## 改了什么

透明参考实现每个 batch slot 都对 W64 内每个候选：

1. 复制不断增长的完整 compiled set；
2. 复制 LRU state；
3. 重复排序同一 query 的 locator cohort；
4. 对每个候选重新扫描所有更老 pending query，判断 B32 feasibility。

增量实现保持相同 floating-point utility 与 tie-break，只做三项等价变换：

- arrival 时一次性缓存 sorted cohort；
- candidate cost-only evaluation 只复制 80-page virtual LRU，不复制 compiled set；
- 利用统一 B、单一 arrival total order 下 oldest pending bypass 最大的不变量：head bypass<32
  时全部 visible 可行，head=32 时只有 head 可行。forced case 仍恢复原实现相同的 protected
  query audit。

selected query 的 sorted-page LRU mutation、build/reload/hit、compiled publication 均未近似。

## 语义等价硬门

五域×burst/Poisson×五 seeds，共 50 个完整 replay。reference 是未修改的 `29448fd`
HardBudgetFrontier；同时运行 transparent-instrumented 与 incremental 两版。

| 检查 | 结果 |
|---|---:|
| 完整 dispatch order / atomic batch | 50/50 exact |
| completion、wait、sojourn tuple | 50/50 exact |
| total work、build/reload/hit、final union | 50/50 exact |
| bypass、forced/protected、B32 violation | 50/50 exact |
| timer wait、batch utilization | 50/50 exact |
| publication trace/hash | 50/50 exact |
| qrel/future-array interface | 无 |

因此 CPU 差异不是通过少服务 query、改变 order、预取或放松公平换来的。

## Deterministic operation proxy

统一 v2 proxy 统计 logical items：events/selection/utility、page probes、feasibility/frontier
comparison、state-copy item 与 cohort-order item。它不是 CPU instruction 数，但两实现粒度
一致，并能解释壁钟结果。

| 域 | burst baseline/opt | Poisson baseline/opt | optimized ops/query（约） | Delay ops/query（约） |
|---|---:|---:|---:|---:|
| HR | 8.24× | 8.43× | 5,689 | 5,647 |
| Finance | 12.50× | 12.77× | 5,678 | 5,637 |
| Computer Science | 8.75× | 8.98× | 5,282 | 5,256 |
| Industrial | 13.13× | 13.53× | 5,597 | 5,553 |
| Pharmaceuticals | 13.41× | 13.70× | 5,823 | 5,776 |

十个 domain×arrival cell 的 ratio 中位 `12.63×`、范围 `8.24×--13.70×`。50 replay
合计分解显示：

- transparent state-copy items：921,335,145；
- incremental state-copy items：65,532,931；
- transparent repeated cohort-order items：16,730,300；
- incremental arrival-time cohort-order items：297,800；
- page probes 两者保持 16,730,300；
- 原 all-older feasibility comparisons 为 16,349,442；incremental 用 625,898 次 head/
  forced-audit frontier comparisons。

所以最大收益来自删除 compiled-set copy；head frontier 与 cohort cache 是第二层收益。

## CPU wall time

注册计时对象必须说清楚：

- `transparent`：与 reference 语义相同、带 v2 counter 的透明实现，是预注册 CPU/内存 gate
  的正式基线；
- `incremental`：优化实现；
- `reference`：未插桩 `29448fd`，只作额外 sensitivity audit，**不用于注册过门**；
- `delay_d32`：同粒度 efficient closest-work comparator。

每个五域×arrival workload 使用 seed `20260804`，small=64 queries、large=完整 trace；每方法
2 warmups+9 repeats，固定轮转，`perf_counter_ns`，计时区间禁用 GC。

### 注册 transparent 基线结果

| 域 | large CPU speedup（两 arrival 均值） | incremental / Delay large CPU | large peak memory ratio |
|---|---:|---:|---:|
| HR | 1.88× | 0.992 | 1.028 |
| Finance | 2.12× | 0.985 | 1.034 |
| Computer Science | 1.86× | 1.012 | 0.989 |
| Industrial | 2.41× | 0.995 | 0.971 |
| Pharmaceuticals | 2.18× | 0.979 | 1.038 |

跨十个 large workload：

- transparent/incremental median speedup：`2.084×`；
- min/max：`1.838× / 2.499×`；
- P95：`2.419×`；
- small incremental/transparent median：`0.639`；最坏 `0.663`，即最差也约 1.51× 加速。

incremental/Delay 的 large CPU 中位接近 1；这比单看 operation proxy 更直接说明 continuous
age/completion scoring 本身没有留下显著控制面税。

### 未插桩 reference sensitivity audit

为了排除 transparent counter 抬高基线时间，额外运行原始 `29448fd`，但不改变注册 gate：

- reference/incremental large CPU median：`2.111×`；范围 `1.781×--2.378×`；
- small incremental/reference median：`0.657`；最坏 `0.726`；
- tracemalloc incremental/reference median：`0.971`。

敏感性结果与正式基线一致，说明加速不是 counter instrumentation 伪影。

## 内存

缓存 sorted cohort 使确定性 retained state-item ratio 中位为 `1.78×`；最大绝对增量 7,280
items，按注册的 8 bytes/item 下界为 58,240 bytes（约 0.056 MiB），远低于 64 MiB。

更重要的是，独立五次 tracemalloc 包含 transient allocation：

- 注册 incremental/transparent peak ratio 中位 `0.971`；
- P95/最大均约 `1.038`；
- incremental 单 replay median peak 范围约 0.49--2.40 MiB。

虽然持久 cohort cache 增加少量状态，删除巨大的 transient compiled copies 抵消了它；内存
门通过。tracemalloc 与 wall time 都是本机经验量，不进入 deterministic JSON。

## 对论文的影响

这轮可以固化成系统实现贡献：

1. Hard policy 可以用 head frontier 给出 Delay-equivalent B32 ordering guarantee，同时保留
   completion+age scoring；
2. candidate cost evaluator 面向 compiled/LRU physical state，但无需为每候选复制 compiled；
3. 五域 exact replay 证明优化没有悄悄改变 anytime quality 或公平；
4. optimized Hard 控制面与 efficient Delay 基本等成本，因此论文可以比较 scoring 效果，
   而不再背负明显的实现低效。

但不能把 `12.63× operation` 写成端到端 RAG/GPU throughput：真实 representation compile
通常更贵，本轮 CPU 加速只针对 Python scheduler+replay control plane。更稳的表述是
“2.08× local single-thread control-plane speedup with exact decisions”，并把 operation proxy
作为机制解释。

## 审计与复现

- 全部 control policy 无 qrel/future arrival 接口；
- exact LRU eviction-before-late-hit、B32 head feasibility 穷举、Delay exact-D32、operation
  counter 与完整 replay equivalence 均有测试；
- deterministic JSON 连续两次 byte-identical；
- deterministic SHA-256：
  `e79633056a8e6cf92cfe00281590bb26acb7cdd7eb3fad61efda5e9efa62c2fe`；
- empirical JSON 是明确标注的非确定性产物，本次 SHA-256：
  `cde38cbf7eb44c04863cb22e638dd9a4b224eb74e4ce373d85e2f4688b558818`。

复现命令：

```bash
PYTHONPATH=. python tools/analyze_incremental_causal_control_plane.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --domain-matrix-root /private/tmp/reprforge-vidore-domain-matrix-v1 \
  --domain-matrix-reference /Users/aura/gpu-systems-incubator/reprforge-worktrees/vidore-domain-matrix/results/diagnostics/vidore-v3-domain-matrix-v1.json \
  --output results/systems/incremental-causal-control-plane-deterministic-v1.json

PYTHONPATH=. python tools/benchmark_incremental_causal_control_plane.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --domain-matrix-root /private/tmp/reprforge-vidore-domain-matrix-v1 \
  --domain-matrix-reference /Users/aura/gpu-systems-incubator/reprforge-worktrees/vidore-domain-matrix/results/diagnostics/vidore-v3-domain-matrix-v1.json \
  --deterministic-result results/systems/incremental-causal-control-plane-deterministic-v1.json \
  --output results/systems/incremental-causal-control-plane-empirical-v1.json
```
