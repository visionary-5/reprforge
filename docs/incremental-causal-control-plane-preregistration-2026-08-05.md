# Incremental causal control plane 预注册

日期：2026-08-05

分支：`exp/incremental-causal-control-plane`

基线：`29448fd`（Causal Hard Frontier）

状态：在实现优化 scheduler 或运行计时前冻结。

## 研究问题与结论边界

上一轮 closest-work 审计表明 HardBudgetFrontier 与 Delay-D32 共享同一个 B32 可行域，主方法
只获得约 1% 的系统指标优势，却在透明 Python 实现中有明显更高的控制操作数。本轮不修改
算法或论文 claim，只回答工程问题：能否在 dispatch 完全不变的前提下，把 hard method 的
控制面实现成增量 physical-plan executor，使它不再为每个 slot/candidate 复制整个 compiled
set、重复排序 cohort、并扫描所有 older-query feasibility？

若优化改变任何选择或成本语义，直接失败。若 exact 但 CPU/operation 没有注册级收益，判
`INCREMENTAL CONTROL PLANE NO-GO`；不能用小 replay 的噪声包装加速。

## 冻结算法与禁止变化

参考方法为提交 `29448fd` 的 `HardBudgetFrontier`。以下全部冻结：

- batch8、visible W64、LRU80、timeout16；
- B32；completion weight 0.75、continuous-age weight 0.25、scale256；
- build/reload/hit unit cost、sorted page access、atomic publication、tie-break；
- 无 prefetch、无 qrel/未来 arrival 接口、相同 arrival/timer event loop；
- candidate union、compiled/cache 生命周期和 quality post-hoc accounting。

优化版只允许：arrival 时缓存 sorted cohort；cost-only evaluation 不复制 compiled set；维护
当前 virtual LRU 的紧凑候选 cost；利用统一 B 下 oldest/head bypass 最大这一不变量，将
all-older feasibility 化为 head skip frontier；在 forced case 中仍恢复与参考实现相同的
protected-query audit。不得近似 LRU、跳过 candidate、改变浮点 utility 或排序。

## exact equivalence 硬门

五域固定为 HR、Finance、Computer Science、Industrial、Pharmaceuticals；burst/Poisson；
seeds `20260804`--`20260808`，共 50 个 full-trace replay cell。输入与 SHA 沿用 causal
hard frontier transfer，不下载数据、不使用 GPU/SSH。

每个 cell 透明 reference、instrumented transparent baseline 和 incremental scheduler 必须
逐项一致：

1. 每个 atomic dispatch batch、完整 dispatch order/hash；
2. completion page/time、wait/sojourn tuple；
3. total work、build/reload/hit、final union；
4. bypass tuple、forced/protected/budget violation；
5. publication trace/hash、timer wait、batch utilization；
6. 无 qrel/future interface 与固定 config。

任一 cell 失败即 `SEMANTIC EQUIVALENCE FAIL`，不报告 deployable speedup。

## deterministic operation proxy

为了避免只比较不同粒度的旧 counter，新增两方法共用的 v2 logical-item proxy：

- arrival/timer/dispatch/selection/utility evaluation 各 1；
- 每个 exact page probe 各 1；
- 每个 feasibility/frontier comparison 各 1；
- evaluation 时每复制/读取一个 compiled 或 LRU state item 各 1；
- 每次排序/缓存 cohort 时每个 cohort item 各 1。

透明 baseline 如实计入每 candidate 的 compiled/LRU copy 与重复 cohort sort；incremental 计入
arrival-time sort cache、LRU-only cost evaluation 和 head frontier。另实现同粒度的 efficient
Delay-D32 作为强控制面基线。operation proxy 是算法实现工作量，不伪装成 CPU instruction、
GPU throughput 或理论复杂度证明。

## CPU wall time 与内存协议

计时使用单 Python process、`time.perf_counter_ns`，每次创建全新 replay/scheduler：

- workload 固定为五域×burst/Poisson×seed `20260804`；
- small trace 为 frozen arrival order 的前 64 queries，large 为完整域 trace；
- 每个 method/workload 先 warmup 2 次，再测 9 次；方法顺序按固定轮转，报告 median/P95；
- 主比较为 end-to-end replay wall time，不能只计优化函数；
- 同时用 `tracemalloc` 独立测 5 次 peak bytes，避免追踪器污染主 timing；
- 记录 Python/platform、query/page/union size。机器有其他负载时只按 ratio 解读，不声称
  production latency。

wall time 与 tracemalloc 天生非确定，写入独立 `*-empirical-v1.json`；exact/operation/protocol
写入连续两次 byte-identical 的 deterministic JSON。内存还保存确定性的 peak retained
cache/state-item count。

## GO 门

只有同时满足以下条件才写 `INCREMENTAL CONTROL PLANE GO`：

1. 50/50 full cells 全部 exact；
2. 五域十个 domain×arrival 聚合 cell 的 deterministic operation baseline/optimized ratio
   中位数 `>=2.0`，**或** large-trace CPU median speedup 的跨 workload 中位数 `>=1.5`；
3. 任一 small workload 的 optimized/baseline median wall time不得 `>1.10`；
4. optimized tracemalloc peak 的跨 workload中位 ratio `<=1.25`，且单 replay额外 retained
   scheduler state `<64 MiB`；
5. optimized operation count 与 efficient Delay-D32 同粒度报告，不隐藏 completion+age 的
   剩余开销。

CPU 与 operation 二者只需一个达到主加速门，但其余回退/内存/exact 门全部必须通过。阈值
不根据结果修改。

## 固定产物

- 本预注册独立提交；
- transparent-instrumented、incremental HardBudgetFrontier、efficient Delay-D32；
- 50-cell exact + deterministic operation JSON；
- small/large wall-time 与 tracemalloc empirical JSON；
- equivalence、head-frontier、exact LRU cost、counter、forbidden interface 测试；
- 完整中文报告、全仓测试、deterministic JSON 连续复跑、clean local commit；不推远程。
