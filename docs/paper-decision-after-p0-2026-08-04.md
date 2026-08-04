# ReprForge P0 反事实后的论文决策

日期：2026-08-04。状态：**停止调度启发式微调；保留系统/测量主线，算法论文需新方法。**

## 一句话判断

ReprForge 已经证明“视觉多向量索引构建过程值得被单独研究”，但没有证明当前 frontier 是足够突破性的最终算法。

现有最强证据是跨五域、跨两种检索器、真实 A100 和 30K-query 控制面规模；最强反证是 HR 无标签选择的 bounded CaGR adaptation 在 Finance 的平均到达—发布时延和计费工作量上同时优于 frontier。统一时间轴又表明方法排名会随负载和质量预算交叉。因此正确的新问题是**异构索引构建中的系统效率—证据质量—尾部公平多目标编译**，而不是继续声称某个单一排序器全面更优。

## P0 结果表

| 实验 | 分支 / 提交 | 决策 | 核心数据 | 含义 |
|---|---|---|---|---|
| EdgeRAG-faithful page cache | `exp/edgerag-faithful-baseline` `8f5b21a` | **CONTINUE** | frontier+同一缓存相对最强 EdgeRAG-page measured encode work 少 16.7%--20.1%，排队 mean 少 29.8%--36.8%，四设置 P95 均更好 | 宽泛缓存不能解释 frontier 全部收益 |
| CaGR-faithful θ=.5 | `exp/cagr-faithful-baseline` `a3cd197` | **条件 GO** | frontier completion pages 少 7.2%--20.9% | 但 98.4%--99.2% group 为 singleton，不能当最强 CaGR 结论 |
| Strong CaGR no-wait | `exp/cagr-strong-adaptation` `7fd795f` | **不可证实 / DOWNGRADE** | 81/81 HR 候选因 Poisson 可见性约束无 deployable selection | 不是 Finance 反例，也不能证明 frontier 胜最强 adaptation |
| Bounded-wait CaGR | `exp/cagr-bounded-wait` `af893b8` | **真实 STOP/DOWNGRADE** | Finance burst/Poisson：CaGR sojourn 2204/1887 vs frontier 2302/2042；work/q 15.829/15.762 vs 16.199/16.477 | 否定“frontier 在系统轴全面优于强 CaGR” |
| Time-aligned quality | `exp/time-aligned-quality` `887eaa2` | **TRADEOFF** | Finance burst elapsed regret frontier 低 2.66%；Poisson bounded CaGR 低 2.55%；曲线交叉 | unique pages、charged work、elapsed time 三轴不可混称 |
| Cost-first locality frontier | `exp/cost-locality-frontier` `3e2d33c` | **NO-GO** | 相对 overlap unit-cost 好 7.8%--20.8%，但相对 page frontier 完成收益约 -0.03%--2.31%，公平更差 | locality 信号有效，但不是第二贡献 |
| Completion-constrained locality | `exp/frontier-constrained-locality` `225be9a` | **NO-DEPLOYABLE** | delta=1 形成均值折中，但 P95 全部失败 | 简单无权重 Pareto key 不足以解决 tail |
| Deadline/oldest override | `exp/deadline-constrained-locality` `b3b4662` | **NO-GO；停止微调** | 两规则 starvation 仍约 21%--23%，P95/pages/sojourn/work 同时失败 | 下一步需要有保证的 deadline 机制，不是再换 tie-break |
| Multi-objective oracle headroom | `exp/multiobjective-oracle-headroom` `ad11bbd` | **注册家族内无安全点** | 60 个 qrel/cost/future-aware 配置中，24 个通过主要目标与 P95、4 个通过 starvation，但交集为 0；最强非安全点的六轴最坏比率为 0.828，starvation 仍为 12.26% | 简单权重扫描存在结构性效率--公平冲突；该结论仅限注册的 greedy oracle 家族 |

## 经独立复核的 Finance 反例

`exp/cagr-bounded-wait` 的 JSON 被另一分支逐字节复跑，SHA256 为
`0a57cca8755ed9d187bf95953471ad10fb31cea26feec791c221af2872efd151`，关键 16 项测试通过。

两方法使用相同 arrival seed、unit service clock、capacity=80、batch=8；build、reload 和 prefetch 都收费 1，hit 为 0。CaGR 的预取不是免费的，仍在两个 Finance 到达模型上获得更低 mean sojourn 和 work/query。批次利用率也几乎相同，因此反例不是 batch 欠填造成。

必须保留的边界：CaGR 带 next-batch prefetch，frontier 没有，所以反例比较的是完整部署 bundle；unit work 不是硬件墙钟；bounded wait 本身贡献很小，主效应来自 fixed-Jaccard locality、prefetch 和 reload-aware service。

## 指标修正

此前文档中“quality--work regret”需要拆成三个明确横轴：

1. **不同已编译页面数**：只统计首次视觉 build，回答每构建一个新页面得到多少质量；
2. **计费工作量**：统计 build、reload、prefetch，回答系统实际做了多少离散工作；
3. **elapsed service clock**：再加入 idle 和显式等待，回答质量何时对到达查询可用。

Frontier 在第 1 轴明显更强；bounded CaGR 在系统 mean/P95 和第 2 轴更强；第 3 轴随 burst/Poisson 反转。任何论文图都必须写清横轴，不能再统称“前沿调度质量更快”。

## 当前能安全写的贡献

1. **问题与测量：** 首次把固定视觉多向量目标下的索引 onboarding 写成查询—页面共享依赖、原子发布和三轴 anytime 评价问题；“首次”仍需按最终 related-work 文案谨慎限定，不能覆盖 EdgeRAG 的在线索引概念。
2. **系统：** ReprForge 实现可恢复、可版本化的异构表示构建和精确增量控制面；30K 查询低于 4.1 秒和 300 MiB。
3. **实证：** 五个 ViDoRe 域、IRPapers、ColPali/ColModernVBERT 和真实 A100 表明构建顺序会显著改变查询完成和质量轨迹。
4. **发现：** EdgeRAG 风格缓存不能消除共享 completion 价值，但强 CaGR locality 能在部分负载反超系统成本；方法排名随计量轴和负载反转。

这些足以支撑系统 measurement、benchmark 或经验型 IR 论文的核心。它们不足以支撑“frontier 是统一最优的新算法”或 ICLR 式方法突破。

## 如果继续冲方法论文，唯一值得做的新方向

停止使用加权启发式或更多 tie-break。新的方法问题应写成显式约束优化：

> 在不可见 qrel 的条件下，最小化 build+reload+prefetch 与 arrival-to-publication sojourn，同时约束每个查询 deadline、随 elapsed time 的证据质量遗憾，以及持久状态容量。

需要的真正新设计至少包含一项：

- 可证明的 per-query deadline / bounded bypass 机制，而不是 oldest quota；
- 构建前可见的页面成本预测及误差鲁棒调度，而不是 exact per-page cost oracle；
- reader/retriever utility 的可迁移预测或风险约束，使质量约束不依赖 qrel；
- 求解或近似带共享资源的多目标 scheduling formulation，并报告 approximation / regret / constraint violation。

最快的验证顺序：先在现有完整 trace 上做离线 oracle，测量同时满足系统成本、time-aligned quality 和 deadline 的可达空间。如果 oracle 都不能明显支配 bounded CaGR/frontier，两目标存在真实不可消除冲突，应停止算法路线；只有 oracle headroom 足够，才值得训练代理或设计近似算法。

该验证现已完成：预注册的 60 点 clairvoyant greedy 家族在 HR 上没有安全候选，因此 Finance 按合同保持封存。最强非安全候选在 burst/Poisson 上把 mean sojourn、charged work 和 elapsed quality regret 全部压到对应端点的 0.72--0.83 倍，P95 也更好，但 starvation 为 12.26%；仅有的 4 个 starvation-safe 候选不使用 completion 信号，最坏主要比率约 1.38。这个结果否定的是“用现有三个信号做简单加权即可得到全面更优策略”，不是对所有调度算法的不可行性证明。

因此下一项方法工作不能再是权重、tie-break 或窗口微调。只有在引入可证明的 per-query service guarantee、显式约束求解，或新的可观测 utility/cost predictor 后，才值得重新开启算法分支；否则应直接固化 measurement/system 论文。

## 投稿建议

- **若目标仍是 9 月 ICLR：** 现在不要直接写 frontier 方法论文。注册 oracle 家族已经没有安全 headroom，应换投稿定位，除非能提出带服务保证的新 formulation，而非继续调参。
- **若接受系统/IR measurement 论文：** 当前资产已经很强，主线可转为“何时先建哪些视觉表示：异构 RAG 索引构建的三轴 benchmark 与系统研究”，frontier、EdgeRAG、CaGR 作为互补政策而非单一 winner。
- **若目标是项目落地：** 默认用 bounded CaGR/locality 获得成本和尾部；需要最早的新页面质量时用 frontier；根据 workload arrival 与 quality budget 选择策略，不声称一个 policy 通吃。

建议保留 ReprForge 名称，但标题改成不预设 winner：

> **ReprForge: Measuring and Compiling the Quality--Locality Frontier of On-Demand Visual Index Construction**
