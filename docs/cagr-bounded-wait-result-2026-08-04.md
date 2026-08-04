# Bounded-wait CaGR-style adaptation 结果

日期：2026-08-04  
分支：`exp/cagr-bounded-wait`  
预注册：`2d098ff`；实现冻结：`39b2818`；等待审计：`9a276f0`；Pareto 范围审计：`01e92a9`  
机器：CPU only，未使用 GPU  
机器可读结果：`results/systems/cagr-bounded-wait-v1.json`

## 一句话结论

本次出现了真正的封存反例：HR 选择出的 bounded CaGR 在 Finance burst 与 Poisson 上，mean arrival-to-publication sojourn 和 charged unit work/query 都优于 current frontier，构成预注册二维 Pareto dominance。因此 Finance gate 为 **STOP/DOWNGRADE**；当前不能再写“frontier beat strongest CaGR-style deployment baseline”。

这与上一轮 `cagr-strong-adaptation` 的结论不同：上一轮是在 HR 无任何 deployable adaptation，属于不可证实；本轮 12/12 候选均通过 HR 部署资格，且唯一配置在 Finance 真正反向胜出。

## HR-only 冻结配置

选择阶段只打开 HR BM25 manifest/runtime；未打开 HR qrel、视觉 runtime 或 Finance。固定 family 为 fixed Jaccard、target group size 16、pool 64、capacity 80、physical batch 8，并允许 cross-group fill。只搜索 12 个 `wait_budget × min_pending` 候选。

冻结配置：

- wait budget：16 unit-time；
- min pending：4 queries；
- selection score：0.7624；
- worst mean ratio：0.7900；
- mean P95-sojourn/FIFO：0.7436；
- 配置 SHA-256：`924d28748c47c7ff0e5cfa87226d631a95cc724db5650abde9425c7a35796311`。

| HR 指标 | burst | Poisson |
|---|---:|---:|
| mean sojourn / FIFO | 0.7481 | 0.7250 |
| unit work/query / FIFO | 0.7867 | 0.7900 |
| P95 sojourn / FIFO | 0.7507 | 0.7364 |
| starvation fraction | 0% | 0% |
| singleton fraction | 0% | 0.92% |
| batch utilization | 99.38% | 95.57% |

全部 12 个候选都满足两种 arrival 下 P95 sojourn≤1.05×FIFO、starvation≤5%、union/query parity；排序完全由预注册 score 决定。

## 有限等待没有作弊

等待和 build/reload 使用同一个 elapsed unit-time clock；build、reload、prefetch 各收费 1，hit 为 0。主动等待推进 sojourn，但不伪装成 charged work。

| 主动 grouping wait（五 seeds 合计） | HR burst | HR Poisson | Finance burst | Finance Poisson |
|---|---:|---:|---:|---:|
| events | 0 | 5 | 0 | 5 |
| total unit-time | 0 | 52.64 | 0 | 52.64 |
| max single wait | 0 | 16 | 0 | 16 |

最大单次等待严格不超过冻结预算 16。所有 query 恰好发布一次，HR union=895、Finance union=1855；starvation 为 0%，最大 younger bypass 分别不超过 62/61（HR）和 60/57（Finance）。

## Finance 主判停结果

| Finance mean | bounded CaGR | frontier | bounded 相对 frontier 的改善 | gate 中 frontier advantage |
|---|---:|---:|---:|---:|
| burst sojourn | 2,204.44 | 2,302.32 | 4.25% | -4.44% |
| burst unit work/query | 15.829 | 16.199 | 2.29% | -2.34% |
| Poisson sojourn | 1,886.81 | 2,041.67 | 7.59% | -8.21% |
| Poisson unit work/query | 15.762 | 16.477 | 4.34% | -4.53% |

gate 的 advantage 以 adaptation 为分母，负值表示 frontier 更差。两种 arrival 下 bounded CaGR 都在两个主部署指标上严格更好，因此：

- `adaptation_system_pareto_dominates_frontier=true`（burst、Poisson）；
- 两项 gate check 均失败；
- final decision：`STOP/DOWNGRADE`。

它不是通过更差尾延迟换均值：bounded CaGR 的 Finance P95 sojourn 为 4,114（burst）和 3,574（Poisson），frontier 为 4,541 和 4,366；frontier/bounded P95 ratio 分别为 1.104 和 1.221。

## 工作量、缓存与预取审计

| Finance 指标 | bounded burst | frontier burst | bounded Poisson | frontier Poisson |
|---|---:|---:|---:|---:|
| cache hit | 30.18% | 19.00% | 30.68% | 17.61% |
| demand build | 8,205 | 9,275 | 8,243 | 9,275 |
| demand reload | 13,368 | 15,753 | 13,178 | 16,182 |
| prefetch build | 1,070 | 0 | 1,032 | 0 |
| prefetch reload | 1,813 | 0 | 1,900 | 0 |
| prefetch precision | 100% | — | 100% | — |
| prefetch wasted / unused work | 0 / 0 | — | 0 / 0 | — |
| batch utilization | 99.04% | 99.04% | 95.62% | 96.09% |

bounded CaGR 的 prefetch 全部计入 unit work；即使如此，需求侧 build/reload 与更高 cache hit 的节省仍超过预取成本。

Cross-group fill 被实现并记录，但本次 selected 配置实际 `cross_group_count=0`、group purity=1.0。原因是主要 logical group 为 16，而 physical batch 为 8，能够整除；不能把结果归因于跨组填充。该机制在这组 trace 上是 inactive design，不是贡献证据。

## bounded wait 本身贡献有多大

no-wait anchor 使用完全相同 fixed-Jaccard/group/prefetch/service-clock，只把 wait budget 设为 0。

| Finance | no-wait sojourn | wait=16 sojourn | no-wait work/query | wait=16 work/query |
|---|---:|---:|---:|---:|
| burst | 2,204.44 | 2,204.44 | 15.829 | 15.829 |
| Poisson | 1,888.83 | 1,886.81 | 15.816 | 15.762 |

所以不能把全部反例包装成“bounded wait 算法”的胜利：

- burst 下没有主动等待，结果与 no-wait 完全一致；
- Poisson 下 wait=16 只比 no-wait 降低约 0.11% mean sojourn、0.34% unit work；
- 主要变化来自统一的真实服务时钟：reload 也消耗时间，因此执行过程中自然积累 backlog，fixed-Jaccard 能形成较完整 group；
- bounded wait 是一个小但可复现的稀疏到达修正，不是主效应。

## 关键代价：质量/新页面进度更差

bounded CaGR 只在本次 gate 的部署轴——sojourn 与 unit work——支配 frontier；它没有在质量排序轴支配 frontier。

| Finance | bounded normalized quality regret | frontier regret | bounded mean completion pages | frontier pages |
|---|---:|---:|---:|---:|
| burst | 0.6538 | 0.5026 | 1,229.54 | 989.76 |
| Poisson | 0.6387 | 0.5132 | 1,228.32 | 1,000.79 |

regret 越低越好。frontier 更积极优先产生新页面和质量增益，因此 quality-work regret 明显更好；代价是较差 cache locality、更多 reload，以及更高 arrival-to-publication 延迟。它还产生约 23% starvation，而 bounded CaGR 为 0%。

这说明当前真正的研究问题不是“哪个单一调度器全局最好”，而是：

> 如何在新证据/质量进度与表示复用/在线完成延迟之间，学习或编译出可控的 Pareto frontier？

## 对论文叙事的影响

以下表述必须删除或降级：

- “frontier 在强 CaGR-style adaptation 下仍有至少 5% 双指标优势”；
- “前沿调度在部署成本与延迟上普遍优于 grouping”；
- 将上一轮 no-deployable 结果当成 CaGR 被击败的证据。

仍被数据支持的表述是：

1. frontier 更快构建新页面、带来更好的 quality-work trajectory；
2. fixed-Jaccard grouping 更有效复用 active representations，在 sojourn 与 unit work 上形成反例；
3. 两者优化的是不同目标，异构索引编译应显式控制质量—局部性 Pareto，而不是只做单轴排序；
4. bounded wait 解决了稀疏到达的可部署性，但其独立增益很小。

最直接的下一方法不是继续调 wait，而是构建 quality-constrained locality scheduler：在最小化 reload/sojourn 时，对新页面进度或预计 quality regret 设置预算；或反过来，在保持 frontier 质量轨迹的条件下最大化 cache locality。新的 gate 应同时包含 sojourn、unit work 和 quality regret，避免在两个目标之间换名宣称胜利。

## 审计与复现

- HR selection opened files 只有 BM25 manifest/runtime，JSON 记录 `oracle_or_visual_files_opened=false`。
- 配置 digest 在 HR full load 与 Finance load 前后保持一致。
- HR/Finance text、visual、qrel 与 manifests 的路径、字节数、SHA-256 均写入 `inputs_after_unseal`。
- JSON 保留 12 个候选完整表、每项五 seeds 的 dispatch order hash、union、sojourn、work、cache、prefetch、group、purity 和 starvation。
- 成本是抽象 unit-time/unit-work，不是 GPU 墙钟、能耗或美元成本；本实验未使用 GPU。
