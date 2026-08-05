# ReprForge / Causal Hard Frontier 最终证据矩阵

日期：2026-08-05

## 最终判断

ReprForge 已经从“异构索引解耦的工程想法”收敛为一个可研究、可复现的系统问题：

> 到达中的 RAG 查询暴露重叠的页面表示需求；构建与重载会改变一个持久、渐进发布的多模态检索索引。系统必须在严格因果信息下优化延迟、物理工作量和随时间可用的证据质量，同时给每个请求明确的服务顺序保证。

当前主方法实例是 `Causal Hard Frontier`：它只接收逐条 arrival/timer 事件、locator cohort 和当前 compiled/LRU 状态，在与 Delay-D32 等价的 bounded-overtaking 可行域内，用 normalized completion density 与 continuous age 选择 batch。

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
| Causal Hard Frontier | 1.000 | 1.000 | 1.000 | 10/10 | 10/10 | 跨域最稳，但透明实现控制操作更多 |

在单一 arrival total order、统一预算和逐 slot 原子选择下，all-older B32 feasibility 与 head skip-D32 feasible set 结构等价。因此必须撤回“新 hard fairness 算法”或“首次 fairness+locality”表述。可比较的差异只是在同一可行域内，completion normalization 与 continuous age scoring 带来约 1% median 增益和更稳的 slowdown tail，同时透明实现的 detailed control operations 约高 75.6%。

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
2. **因果系统策略：** compiled/LRU state-aware completion+age policy，在 Delay-equivalent bounded-overtaking 可行域中运行；不使用 qrel、未来请求或 EOS。
3. **系统与机制证据：** 原子持久发布、精确重放、五域三轴/P99、private-pages 反事实、fair-locality closest work 和 A100 page-cost robustness。
4. **经验发现：** completion-oriented quality、locality-oriented cost 和 per-query service contract 形成真实 Pareto；指标横轴和 fairness denominator 会改变方法排名。

## 禁止写的过强主张

- 首次异构索引、首次按需索引或首次 cheap locator + vision on demand；
- 首次 query grouping、首次 fairness+locality 或新的 hard bypass；
- 相对一个单一 baseline 在所有轴 Pareto 支配；
- 所有用户子群或单 query max 都更公平；
- 答案更早正确；
- 已验证跨 retriever 的 Causal Hard Frontier；
- A100 per-page additive proxy 等于真实并发 wall-clock。

## 投稿前只剩三项高价值工作

1. **同一 B32 可行域的单变量 scoring 消融：** Delay locality、completion-only、completion+continuous age，解释约 1% 差异究竟来自哪里；如果差异仍很小，算法独立贡献降级为系统 policy。
2. **控制面与真实 GPU 校准：** 增量 marginal-cost cache/protected-prefix，报告 scheduler CPU 占真实 build/reload wall-clock 的比例；至少一个 HR/Finance arrival trace 重跑三次。
3. **跨 retriever 原始 trace：** 在已有 ColModernVBERT artifact 所在机器做 CPU-only B32 replay；不重新编码，先验证 causal transfer，再决定是否做 GPU。

若这三项来不及，论文仍可按系统/IR measurement+method 主线固化；不应为了 ICLR 叙事继续发明新模块或重调 B32。

## 主要本地分支

| 分支 | 结果提交 | 角色 |
|---|---|---|
| `exp/causal-hard-frontier` | `29448fd` | 主方法与五域矩阵 |
| `exp/hard-fair-joint-tail` | `a1291fe` | pooled joint tail |
| `exp/counterfactual-slowdown` | `e646106` | 公平指标校正 |
| `exp/fair-locality-baselines` | `ec900ca` | Delay/DLPM/max-wait closest work |
| `exp/dependency-structure-ablation` | `01c0bd1` | 共享依赖机制识别 |
| `exp/causal-cost-robustness` | `a9cd66d` | A100 page-cost 噪声与漂移 |

所有分支均为本地分支，未推送远程。
