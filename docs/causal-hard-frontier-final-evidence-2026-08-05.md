# ReprForge / Causal Hard Frontier 最终证据矩阵

日期：2026-08-05

## 最终判断

ReprForge 已经从“异构索引解耦的工程想法”收敛为一个可研究、可复现的系统问题：

> 到达中的 RAG 查询暴露重叠的页面表示需求；构建与重载会改变一个持久、渐进发布的多模态检索索引。系统必须在严格因果信息下优化延迟、物理工作量和随时间可用的证据质量，同时给每个请求明确的服务顺序保证。

当前主方法实例是 `Causal Hard Frontier`：它只接收逐条 arrival/timer 事件、locator cohort 和当前 compiled/LRU 状态，在与 Delay-D32 等价的 bounded-overtaking 可行域内，用 normalized completion density 与 continuous age 选择 batch。这里的 completion+age 是冻结的系统策略，不是论文要声称的新评分算法。

它已经达到 **replay 级系统方法候选**，但还不是“全新的公平调度算法”。最强 closest work Delay-D32 只落后约 1%，因此论文的新颖性必须来自在线表示编译的状态语义、物理质量前沿、因果系统实现与跨域机制证据的组合。

## 五域主矩阵

下表均为冻结 `hard_budget_frontier` 的 ratio，越低越好。Mean sojourn 与 work 对 HR-selected bounded CaGR；elapsed quality regret 对 frontier；P99 对两者中较好的 endpoint。

| Domain | Arrival | Mean sojourn | Work/query | Elapsed regret | P99 | 至少一轴改善 5% |
|---|---|---:|---:|---:|---:|---|
| HR | Burst | 0.932 | 0.950 | **0.824** | 0.949 | 是 |
| HR | Poisson | 0.923 | 0.956 | **0.851** | 0.940 | 是 |
| Finance | Burst | **0.919** | **0.942** | 0.995 | 0.941 | 是 |
| Finance | Poisson | **0.922** | **0.949** | 0.963 | 0.940 | 是 |
| Computer Science | Burst | **0.911** | **0.943** | **0.944** | 0.936 | 是 |
| Computer Science | Poisson | **0.921** | **0.946** | **0.912** | 0.928 | 是 |
| Industrial | Burst | 0.953 | 0.975 | 0.987 | 0.974 | 否；最大改善 4.713% |
| Industrial | Poisson | **0.949** | 0.971 | **0.953** | 0.968 | 是 |
| Pharmaceuticals | Burst | **0.923** | **0.947** | 0.974 | 0.953 | 是 |
| Pharmaceuticals | Poisson | **0.923** | **0.949** | **0.912** | 0.943 | 是 |
| **跨 cell median** | — | **0.923** | **0.949** | **0.948** | — | **9/10** |

所有 cell final union/query parity，max younger bypass=32，budget violation=0；10/10 P99 优于冻结安全 endpoint。

## 从想法到方法的证据链

| 研究门 | 结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 60 点 soft oracle | 24 个 primary/P95 合格、4 个 bypass 合格，交集 0 | 软 Lagrangian 无法表达 hard service contract | 不是全局不可能性证明 |
| Hard budget B={8,16,32,64} | HR 唯一 B32；Finance 两 arrival 均 GO | 存在非平凡“过紧--可行--过松”区间 | bypass=0 不等于用户公平 |
| 独立 causal API | HR/Finance 20/20 cell 全 tuple exact | 结果不依赖 qrel、未来 trace 或 EOS | 仍是 replay，不是真实并发系统 |
| Joint user tail | HR/Finance 4/4，五域 P95/P99 10/10 | pooled sojourn 与 slowdown tail 可迁移 | 不支配所有单 query max |
| Counterfactual slowdown | HR Q4 P99 从 1.814/1.736 修正为 0.945/0.896 | 原 Q4 反转是 policy-dependent 分母伪影 | strict max 仍有单 query 1.167× 反例 |
| Five-domain transfer | 三主轴无 cell 超过 1.0；9/10 达 5% | 跨域方向稳定 | Industrial burst 效果小 |

## Closest-work 主表

以下 ratio 以 Causal Hard Frontier 为 1；低于 1 表示 baseline 更好。

| Method | Median mean | Median work | Median elapsed regret | B32-safe | P99 sojourn+slowdown safe | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Delay-D32 | 1.009 | 1.007 | 1.011 | 10/10 | 6/10 | 极危险近邻；7/10 三轴在 2% 内 |
| DLPM query adaptation | 1.038 | 1.005 | 1.012 | 0/10 | 5/10 | one-shot query 破坏原 per-client deficit 语义 |
| Max-wait overlap | 1.298 | 1.237 | 1.228 | 10/10 | 0/10 | 时间 cap 不能替代 completion utility |
| Causal Hard Frontier | 1.000 | 1.000 | 1.000 | 10/10 | 10/10 | 跨域最稳；透明实现开销已由 exact 增量版关闭 |

在单一 arrival total order、统一预算和逐 slot 原子选择下，all-older B32 feasibility 与 head skip-D32 feasible set 结构等价。因此必须撤回“新 hard fairness 算法”或“首次 fairness+locality”表述。后续严格单变量消融进一步否定了评分新意：冻结 completion+age 相对每个 cell 最强的 Delay、completion-only 或 age-only，只在 3/10 个 domain×arrival cell 让 mean sojourn、work/query 和 elapsed regret 同向改善；三项比率的跨 cell median geometric mean 为 1.0009，P99 明显改善为 0/10。它应降级为合理的 system policy，而不是独立 contribution。

## 第四轮决策补充

| 问题 | 分支 / 提交 | 注册结论 | 关键数据 | 对主线的影响 |
|---|---|---|---|---|
| 同一 B32 可行域内的评分 | `exp/b32-scoring-ablation` `f5b04c1` | **SCORING NO-GO** | 三主指标同向 3/10；median geometric mean 1.0009；明确 P99 win 0/10 | completion+age 只保留为冻结策略；exact build+reload 状态和 normalization 作为组件证据 |
| 精确增量控制面 | `exp/incremental-causal-control-plane` `b375224` | **GO** | 五域 50/50 exact；operation proxy 12.63×；单进程 CPU 2.084×；peak memory ratio 0.971 | 删除透明实现开销后，Hard 与 efficient Delay 控制面接近；这是系统实现贡献 |
| 工作负载物理计划选择 | `exp/workload-physical-plan-selector` `51014c3` | **ORACLE GO / DEPLOYABLE NO-GO** | oracle 综合比 0.961；LODO selector 1.001 且 24 次 SLO violation | 当前 locator 结构不能安全预测下一 episode 的 evidence value；不升格为方法 |
| 长周期重优化 | `exp/adaptive-compile-reoptimization` `44c81d1` | **NO-GO** | 带成本 oracle 仅改善 0.227%；免费 oracle 也仅 0.637% | 当前五计划家族没有足够 reoptimization headroom；停止做 trigger/learner |
| 多级表示编译 | `exp/multilevel-representation-compiler` `88bf45d` | **ARTIFACT NO-GO / ORACLE HEADROOM** | Finance/Industrial 三路 query oracle nDCG@10 分别高 0.1105/0.1227 | 非单调多表示有真实质量空间，但缺 mixed-state activation、逐 item build/reload 和同 workload 成本，不能称动态 compiler 结果 |

增量实现没有改变 dispatch、B32、W64、LRU、成本或发布轨迹。它通过 arrival-time cohort cache、只复制 80-page virtual LRU，以及等价的 head-feasibility frontier 消除重复工作。optimized Hard 相对 efficient Delay 的 operation 仅高约 0.75%，large CPU 比接近 1，因此此前约 75.6% 的控制操作差距主要是透明实现问题，不是 completion+age 固有代价。

## 机制消融

| 条件 | Median sojourn gain | Median work gain | Median elapsed-regret gain | 解释 |
|---|---:|---:|---:|---|
| 原始依赖图 | 7.69% | 5.08% | 5.16% | 完整共享结构 |
| 保持 query/page degree 的随机换边 | 4.66% | 3.71% | 10.82% | 度数/热度足以产生机会，精确拓扑调节幅度 |
| 每 query 私有页面 | -0.10% | **0.00%** | -0.13% | 跨查询共享消失后收益消失 |
| 每批清 compiled+cache | 相对 bounded 仍正，但相对 overlap 反转 | 同左 | elapsed 反转 | baseline-state interaction 很强，不能只看单一 reference |

清除持久状态会让 hard method 的绝对 sojourn/work 增加约 7.7%/5.7%。但在 build=reload=1 的单位成本下，清全部状态与只清 active cache 的总 work 相同，不能据此区分编译持久化和缓存持久化。

## 成本可观测性

隐藏真值使用冻结的 A100 per-page ColPali encode time 可加 profile；scheduler 只见预测值。

| 预测条件 | 相对 perfect 的 median mean/P95/P99/work/regret | 判断 |
|---|---|---|
| Unit page count | 0.980 / 0.975 / 0.981 / 0.982 / 0.978 | 不依赖 exact cost；myopic perfect 不是全局上界 |
| Unbiased noise CV=0.5 | 1.036 / 1.032 / 1.034 / 1.025 / 1.030 | 注册鲁棒门通过；最坏 cell ≤1.055 |
| Noise CV=1.0 | 1.089 / 1.076 / 1.077 / 1.060 / 1.050 | 相对 bounded 的优势基本被吃掉 |
| Expensive-tail underestimation | 1.053 / — / — / 1.026 / 1.031 | domain drift 风险；单 cell regret 最坏 1.157 |
| CV1 winsorized | 1.057 / 1.054 / 1.055 / 1.041 / 1.044 | 明显恢复，但不是每 cell 每轴都恢复 |

这证明方法在 unit 或中等噪声预测下不依赖 cost oracle；仍不能替代真实 reload I/O、batch overlap、GPU 并发干扰和 wall-clock 测量。

## 可以安全写的贡献

1. **问题与评价：** 在线渐进式多模态索引编译；到达查询暴露重叠表示需求，完成 build 持久改变后续可用证据；评价分开 elapsed、charged work 和 unique published pages。
2. **因果编译合同：** policy 只能读取已到达 locator demand 与当前 compiled/LRU physical state；不使用 qrel、未来请求或 EOS，并在 Delay-equivalent B32 可行域中运行。
3. **精确增量控制面：** 在 50/50 完整 trace 保持 dispatch、work、tail 和 publication exact 的同时，把透明控制面 operation 降低 12.63×、本地单线程时间降低 2.084×，且内存不回退。
4. **系统与机制证据：** 原子持久发布、五域三轴/P99、private-pages 反事实、exact build+reload 状态消融、closest work 和 A100 page-cost robustness。
5. **经验发现：** completion-oriented quality、locality-oriented cost 和 per-query service contract 形成真实 Pareto；指标横轴和 fairness denominator 会改变方法排名。

## 禁止写的过强主张

- 首次异构索引、首次按需索引或首次 cheap locator + vision on demand；
- 首次 query grouping、首次 fairness+locality 或新的 hard bypass；
- 相对一个单一 baseline 在所有轴 Pareto 支配；
- 所有用户子群或单 query max 都更公平；
- 答案更早正确；
- 已验证跨 retriever 的 Causal Hard Frontier；
- A100 per-page additive proxy 等于真实并发 wall-clock。

## 下一轮只做最小物理闭环

评分消融和增量控制面已经完成，不再搜索 B、completion/age 权重、tie-break、selector 或 reoptimization trigger。投稿前最有价值的工作缩成两项：

1. **同 workload 的真实硬件成本：** 测量实际 batch 下的 build、active-LRU reload/H2D、命中和并发 overlap，把当前 unit/additive proxy 校准成可重放的物理成本表，并报告 scheduler CPU 占端到端 wall-clock 的比例。
2. **MMDocIR 多级激活 artifact：** 为每个 query 保存 locator activation IDs、pool/full 混合执行结果、逐 item build/reload/bytes 和表示 lineage；先跑 static/dynamic oracle，只有两种文档角色都相对 full-eager、transient 和 GDSF 在 matched nDCG@10 下节省至少 10% charged cost，才继续设计 compiler。

跨 retriever 原始 B32 trace 仍是边界，但不再优先于上述物理闭环。论文可以按系统/IR measurement+method 主线固化；不应为了 ICLR 叙事继续发明评分公式或重调 B32。

## 主要本地分支

| 分支 | 结果提交 | 角色 |
|---|---|---|
| `exp/causal-hard-frontier` | `29448fd` | 主方法与五域矩阵 |
| `exp/hard-fair-joint-tail` | `a1291fe` | pooled joint tail |
| `exp/counterfactual-slowdown` | `e646106` | 公平指标校正 |
| `exp/fair-locality-baselines` | `ec900ca` | Delay/DLPM/max-wait closest work |
| `exp/dependency-structure-ablation` | `01c0bd1` | 共享依赖机制识别 |
| `exp/causal-cost-robustness` | `a9cd66d` | A100 page-cost 噪声与漂移 |
| `exp/b32-scoring-ablation` | `f5b04c1` | 评分独立贡献 NO-GO |
| `exp/incremental-causal-control-plane` | `b375224` | exact 增量控制面 GO |
| `exp/workload-physical-plan-selector` | `51014c3` | oracle 有空间、可部署选择器 NO-GO |
| `exp/adaptive-compile-reoptimization` | `44c81d1` | 长周期重优化 NO-GO |
| `exp/multilevel-representation-compiler` | `88bf45d` | 多级真实 surface headroom 与 artifact 缺口 |

所有分支均为本地分支，未推送远程。
