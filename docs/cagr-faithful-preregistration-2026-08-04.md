# CaGR-RAG 忠实基线预注册

日期：2026-08-04  
分支：`exp/cagr-faithful-baseline`  
状态：在产生本分支结果前冻结。

## 研究问题与判停门

现有 `overlap_only` 只是“先选全局 overlap degree 最大的种子，再贪心填满请求批次”，不是 [CaGR-RAG](https://arxiv.org/abs/2505.01164) 的算法。本实验回答：在同一 HR/Finance、BM25 Top-20 查询访问集、W64 有界到达和 request batch 8 下，当前 frontier 相比 CaGR-RAG Algorithm 1 的阈值分组、组连续执行与下一组预取，是否仍有独立调度收益？

论文主调度贡献只有在 HR/Finance × burst/Poisson 四个主设置全部满足以下条件时 GO：

1. frontier 相对 `cagr_faithful` 的 mean completion page-work 或冻结 measured-cost replay 至少改善 5%；
2. frontier 的归一化质量遗憾不高于 CaGR-RAG；等价地，quality--work AUC 不低于 CaGR-RAG（数值容差 `1e-12`）；
3. frontier 的 P95 completion page-work 与 P95 measured cost 均不得比 CaGR-RAG 高超过 5%；
4. 最终候选并集、最终完成查询和最终冻结质量完全相同。

任一领域/到达模型不满足即 **STOP/retitle scheduler main claim**。不得用跨设置平均、阈值后调或单独挑页工作/成本中较有利者恢复 GO。

CPU 回放先执行。只有四个设置全部通过，并且 measured-cost 模型的输入来自已冻结真实测量，才允许申请真实 GPU；CPU NO-GO 时不使用组内共享 GPU。

## 论文算法的冻结解释

依据 [CaGR-RAG HTML 第 3.2 节与 Algorithm 1](https://arxiv.org/html/2505.01164v1)，主实现冻结如下：

- 查询访问集 `C(q)`：该查询 BM25 Top-20 的**精确页面 ID 集合**；不使用文本、query embedding、视觉分数或 qrel。
- 相似度：`J(A,B)=|A∩B|/|A∪B|`。
- 阈值：论文评测值 `theta=0.5`。
- 分组池：每次只看已经到达且 pending 的最老至多 `W=64` 个查询。论文使用随机 20–100 查询 batch；64 是当前冻结在线可见窗口内的主适配。诊断池大小固定为 20、40，不进入 GO 判定。
- 分组过程：按原始到达 rank 扫描查询；按 group 创建次序扫描已有 group。若当前查询与该 group **任一成员**的 Jaccard 至少 0.5，则加入第一个满足条件的 group，否则新建 group。
- 歧义处理：论文 Equation 3 写成 group 内全体成员满足阈值，但 Algorithm 1 第 10 行使用 `max(J)>=theta`。主实现遵循可执行伪代码的 `max` 规则；`all` 规则只做敏感性诊断，不能替换主结果。
- group 大小：不设人为上限；记录实际 count、mean/P50/P95/max membership。group 次序为首次创建次序，组内保持到达次序。
- 执行：一个分组池生成的 group 依次完整执行，禁止在两个 group 间交错。物理请求发布仍使用最大 8 个 query 的 atomic request batch；group 不足 8 时不从下一 group 补齐，以保留“group-contiguous”语义并记录 batch utilization。所有策略使用相同最大 request batch 8。
- 重分组：当前分组池全部执行后，从届时最老至多 64 个 arrived/pending 查询重新分组。执行中的新到达不可被当前 plan 观察。
- opportunistic prefetch：完成 group `i` 的最后一个请求批次后，只预取 group `i+1` 第一条查询的 Top-20 页面。已经 active-cache resident 的页面不重复预取。

## 状态、缓存与成本适配

CaGR-RAG 原系统加载的是已离线构建的 IVF 簇；ReprForge 构建的是尚不存在的视觉页表示。为避免把二者混为一谈，回放维护两个状态：

1. `compiled`：已经构建并持久存在的视觉页面集合，不淘汰；它决定 primary completion page-work。所有策略共享相同语义。
2. `active_cache`：当前可直接访问的页面表示，确定性 LRU，容量 80 页。选择 80 是论文 `cache_entries / nprobe = 40/10 = 4` 的比例适配到 Top-20 访问集；所有策略容量相同。诊断容量 40、160 不进入 GO 判定。

访问一个页面时：

- 未在 `compiled`：支付一次 build，写入 compiled 并放入 active cache；
- 已 compiled 但不在 active cache：支付一次 reload 并放入 active cache；
- 已在 active cache：记一次 cache hit，刷新 LRU。

预取也遵循相同状态转移，但必须单独记录：prefetch pages、prefetch build/reload cost、在被需求访问前至少命中一次的 useful prefetch、未命中即淘汰或流结束的 wasted prefetch。不能把预取工作从总工作中删除；只有冻结的异步 measured-cost 模型可以把不超过当前批次可重叠预算的预取延迟从关键路径隐藏，原始工作量仍完整报告。

primary page-work 指 `compiled` 的累计唯一页面数，和现有实验一致。active cache 仅影响 cache hit 与 measured-cost replay，不能让已构建页再次计入 primary page-work。

## 冻结 workload 与方法

- 数据：current HR、Finance manifests；不得换 legacy flat traces。
- candidate K：20。
- request batch：最大 8；除 CaGR group 边界外，其他策略保持原有最大 8 语义。必须同时报告实际 batch utilization，防止 CaGR 因忠实边界而被隐藏地减少资源配额。
- 到达窗口：W64。
- 到达顺序：种子 `20260804`–`20260808` 的五个 permutation。
- burst：32 个查询一批，每 64 page-work 单位到达。
- Poisson：平均间隔 4 page-work，按原冻结 seed 产生。
- 方法：FIFO、当前 `overlap_only`、causal history/static popularity、`cagr_faithful`、当前 frontier；`frontier_prefetch` 仅在预取完全不使用 visual/qrel 且共享同一资源模型时作为诊断，不得代替原 frontier 通过 gate。
- 禁止调度器观察：qrel、视觉分数、答案结果、当前/未来 quality gain、未到达查询。

`static_popularity` 在在线主表中必须是 causal history popularity（优先级只由更早到达查询冻结）；如果另列 full-stream static popularity，必须标成 offline diagnostic。

## 指标

每个方法、领域、到达模型汇总五个 permutation：

- mean/P50/P95/max completion page-work；
- mean/P50/P95 measured completion cost 与 wait/sojourn cost；
- normalized quality regret、quality--work AUC；qrel/visual surface 只在决策后计量；
- active-cache hit fraction、demand build/reload 数；
- prefetch precision、useful/wasted pages、unused prefetch work、可隐藏与未隐藏成本；
- starvation fraction 与 max younger bypass，沿用“至少 W 个更年轻查询先执行”的定义；
- group count/size、request-batch utilization、最终 union parity、顺序/输入 digest。

## 冻结 measured-cost replay

CPU 回放必须同时给两类成本：

1. `unit-work`：build=1 page unit，reload=1 page unit，cache hit=0；这是无硬件假设的保守压力测试，不可称墙钟。
2. `measured-profile`：只允许读取仓库中已存在且带 provenance 的 A100 测量，冻结每域 build-ms/page、reload-ms/page、request-batch overhead 和可重叠预取预算。若现有 artifact 无法辨识 reload 或 batch-level 参数，则该字段明确记为 unavailable，GO 只能由 primary completion page-work 决定，不能拟合本次结果。

不得从 CaGR 结果反推或调优成本参数。真实 GPU 只在 CPU gate 通过后对冻结顺序执行，且不得打扰其他组员进程。

## 已知 fidelity gaps

- 页面 Top-20 是 candidate access set，不是 IVF 的 nprobe=10 cluster set；Jaccard 分布可能更稀疏。
- 原论文优化磁盘读取与 cache locality，本实验主工作是视觉表示构建；两层状态模型显式区分 build 和 reload，但 CPU 仍不是 NVMe/FAISS 实测。
- 原论文 batch 在 20–100 间随机，本实验因现有在线证据边界固定 W64，并保留 20/40 诊断。
- 原论文逐查询搜索；ReprForge 原子发布最多 8 个 query。主实现不跨 group 填 batch，忠实保留 group boundary，但可能降低 request-batch utilization。
- 截至 2026-08-04，定向检索未定位到作者官方 CaGR-RAG 代码仓库；实现依据论文伪代码，不能标为作者实现。

这些 gap 必须在最终报告中逐项保留，无论结果正负。
