# ReprForge 论文证据矩阵

日期：2026-08-04。用途：用研究问题和反事实基线组织实验，避免为了“表很大”而重复跑不能改变论文判断的组合。

## 突破性结果的最低标准

ReprForge 只有同时满足下面五点，才能从“有效系统机制”提升为强论文故事：

1. **跨数据成立**：至少三个完整语料检索域，不能只在 HR 调参后同时报告 HR；
2. **跨表示成立**：至少两种视觉检索表示，排除 ColPali-v1.1 特有批次或分数性质；
3. **下游成立**：更早检索到证据能转化为更早且持续正确的答案；
4. **强反事实成立**：必须比较未来热门度、重叠分组、最短剩余工作、离线工作下界、瞬时不持久方案和容量缓存；
5. **真实系统成立**：冻结方法后报告真实 GPU 时间、P50/P95、构建页数、持久字节和最终质量，而不只报告模拟页工作量。

当前第 1、2 点和第 4 点的大部分已经满足；第 5 点在控制面规模和两域真实 GPU 上部分满足。第 3 点经三种答案协议严格 NO-GO，因此当前只能定位为检索/索引系统论文。最近工作审计还增加了两条优先级最高的 novelty gate：EdgeRAG-faithful 成本缓存和 CaGR-RAG-faithful 分组/预取。

## 研究问题矩阵

| 研究问题 | 最小数据集 | 必须比较 | 主指标 | 当前状态 | 论文判定作用 |
|---|---|---|---|---|---|
| RQ1 渐进构建是否比完整预构建更早可用 | HR、Finance、IRPapers | BM25、完整视觉、静态融合、临时视觉不持久、渐进持久 | 最终质量、构建页/秒、T50/T90、质量遗憾 | HR/Finance 有真实 A100；IRPapers 有冻结回放 | 支撑问题价值和端到端 Pareto |
| RQ2 前沿收益是否只是简单排序 | HR、Finance、IRPapers | FIFO、随机、热门度、重叠、最短缺失、仅复用、离线工作贪心 | mean/P50/P95 完成工作、AUC、距离线下界 | **GO** | 当前最强方法证据 |
| RQ3 看不到完整未来时是否仍有效 | HR、Finance；burst/Poisson | W=1/8/16/32/64/full、历史热门度、硬/软公平 | 收益保留、等待尾部、饥饿、AUC | W64 GO；W32 和软公平 NO-GO | 限定高并发/批量适用范围 |
| RQ4 状态是否能跨会话复用且受限容量可控 | HR→Finance/漂移会话、IRPapers | 无缓存、LRU、LFU、GDSF、Belady、frontier+cache | 重编码、命中率、质量遗憾、P95、字节 | 热启动贡献 **NO-GO**；有限缓存下 frontier 仍稳健 | 缓存是系统功能，不列独立贡献 |
| RQ5 检索提前是否转成答案提前 | IRPapers 180 proxy、24/8 reader pilot、48 held-out | 初始文本、FIFO、热门度、前沿、结构化引用、盲化语义 judge | 首次持续正确、最终准确、正确→错误、引用页 | 三层答案实验 **NO-GO** | 当前不能称答案级 RAG 贡献 |
| RQ6 是否迁移到现代视觉检索器 | HR+Finance | ColPali-v1.1、ColModernVBERT | 检索质量、union/reuse、调度工作、GPU时间/字节 | 跨表示 **GO** | 排除底层模型偶然性 |
| RQ7 是否在大规模语料仍有意义 | 合成到 338K pages / 30K queries；后续真实大语料 | naive exact、增量 exact、全预构建、两阶段瞬时级联 | 控制面时间/RSS、GPU小时、索引字节、吞吐 | 控制面规模 **GO**；真实 338K 编码仍未做 | 证明算法可扩展，但不能替代真实大语料 |
| RQ8 发布修订是否安全 | HR、Finance | 全发布、不发布、固定规则、交叉拟合、oracle | 有害修订、正收益保留、CVaR | 严格 NO-GO | 作为局限，不列主贡献 |
| RQ9 是否只是 EdgeRAG 缓存或 CaGR 分组 | HR、Finance，W64 burst/Poisson | EdgeRAG-faithful 成本缓存、CaGR-faithful/strong/bounded 分组预取、frontier | mean/P95 sojourn、charged work、命中、预取 | EdgeRAG **CONTINUE**；bounded CaGR **STOP/DOWNGRADE** | Frontier 不能再作为全面更优的主算法 |
| RQ10 是否存在同时保质量与局部性的简单调度 | HR 选择、Finance 冻结 | cost-first、completion-constrained、deadline override、60 点 clairvoyant greedy oracle | 三轴 regret、P95、starvation、constraint violation | 三个启发式和注册 oracle 家族均 **NO-GO**；oracle 的 24 个 primary/P95 合格点与 4 个 starvation 合格点无交集 | 停止启发式微调；新方法必须引入显式服务保证或约束求解 |
| RQ11 starvation 指标是否代表用户等待公平 | HR、Finance；burst/Poisson | frontier、bounded CaGR、oracle_15、deadline-only | bypass 阈值、absolute sojourn、per-demand slowdown、tail-label overlap | **bypass-only NO-GO**；16 cells 均未同时强匹配两个 extreme tail，F1 0.013--0.450 | 论文不得把 starvation=0 等同用户公平；采用 joint sojourn+slowdown |
| RQ12 硬顺序预算能否保留多目标余量 | HR 选择、Finance 冻结；burst/Poisson | B=8/16/32/64、bounded CaGR、frontier | mean/P95 sojourn、work/query、elapsed regret、forced slots、budget violation | **GO**；HR 唯一 B32，Finance sojourn -7.8%--8.1%、work -5.1%--5.8%，elapsed regret 不劣，违规 0 | 当前主方法候选；仍需 causal、joint-tail、跨域与 GPU 验证 |

## 数据集分工

| 数据集 | 在论文里负责什么 | 不能证明什么 |
|---|---|---|
| ViDoRe v3 HR | 开发域、真实 A100、复杂企业页面 | 不能单独证明迁移 |
| ViDoRe v3 Finance | 第一冻结迁移域、真实 A100 | 和 HR 使用同一模型，不能证明跨表示 |
| IRPapers | 独立完整语料、文本/视觉互补、180 个 reference answer | 单 gold Recall@5 不能替代 graded nDCG 或复杂多页回答 |
| ViDoRe v3 其他域 | 规模化跨域与 reference-answer 链路 | 没有自然时间戳，不能伪装成真实在线轨迹 |
| MIRACL-VISION | 338K 级多语言规模压力 | 渲染 Wikipedia 且经过 hard-negative 过滤，不代表生产原始分布 |
| M3DocVQA/MMDocRAG | 多页答案与引用正确性 | 单独不能证明完整语料索引效率 |

## 主表与附录组织

### 主表 A：最终 Pareto

每个数据集报告最终检索/答案质量、不同视觉页面数、视觉编码秒、持久字节、总时间。所有方法必须在相同语料和相同 reader 下比较。

### 主表 B：随构建过程

报告归一化质量遗憾、持续 T90、完成工作 P50/P95、25%/50% 预算质量。质量曲线非单调时使用“持续达到”，不能用第一次短暂越线。

### 主表 C：机制反事实

报告 FIFO、热门度、重叠、最短缺失、前沿和离线工作贪心。主结论以 mean work + quality-regret 的二维 Pareto 表达，不用单一加权总分掩盖取舍。

### 主表 D：答案轨迹

报告首次持续正确时间、最终答案正确率、引用证据正确率和正确→错误修订。gold page 到达只算 evidence oracle/proxy，必须与真实 reader 生成分开。

附录放 K×batch×window×capacity 全网格、五个查询排列、置信区间、P95/max、数值扰动和全部负结果。

## 统计与泄漏纪律

- 方法与阈值只在明确开发域选择；Finance、IRPapers 或新域作为冻结迁移；
- “冻结迁移”表示选择代码不读取该域且配置不再改变，不自动等同于研究者从未见过历史 endpoint；已有结果必须在 provenance 中披露；
- 调度时禁止 qrel、visual score、真实质量增益和答案正确性；
- 查询级质量用配对 bootstrap；到达模型至少五个固定排列；GPU 墙钟至少三次，报告中位数和范围；
- 所有顺序必须验证最终候选并集和冻结分数语义一致；
- 后验发现的好参数只作为 headroom，不得替换冻结结果；
- 新 GPU 执行必须先通过 CPU 页工作 gate，并记录模型 revision、数据 SHA、卡型、批次和并发占用。

## 当前最有希望的论文表述

> 给定固定的视觉多向量目标表示，廉价定位器暴露查询—页面依赖图。ReprForge 编译并原子发布共享页面状态，同时分开测量新页面构建、缓存/reload 工作和真实 elapsed quality。实验揭示 completion-oriented frontier 与 locality-oriented grouping 在不同负载下形成不可忽略的系统效率—证据质量 Pareto。

跨表示门已经由 [ModernVBERT/ColModernVBERT](https://arxiv.org/abs/2510.01149) 在 HR/Finance 上通过；五个 ViDoRe v3 域也给出一致工作收益。宽泛的在线构建表述必须让位于 [EdgeRAG](https://arxiv.org/abs/2412.21023)。忠实和增强的 [CaGR-RAG](https://arxiv.org/abs/2505.01164) 对照进一步证明：frontier 在 unique-page 质量效率上更强，但 bounded locality 在 Finance 系统轴反超；本文不能声称单一 scheduler 全面更优。答案级门也失败，因此不声称更早答对。

最后的 clairvoyant headroom probe 进一步限定了方法空间：在预注册的 60 个 qrel、精确成本和有限未来到达可见的 greedy 配置中，没有一个同时通过主要目标、P95 和 starvation 门。它支持“质量--局部性--公平是实质性的多目标边界”这一测量结论，但不构成全局最优性或数学不可能性证明。
