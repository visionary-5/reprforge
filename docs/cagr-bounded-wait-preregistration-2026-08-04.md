# Bounded-wait CaGR-style adaptation 预注册

日期：2026-08-04  
分支：`exp/cagr-bounded-wait`  
状态：在实现、HR 选择或 Finance 开封前冻结。

## 研究问题

上一轮 strong-adaptation 的 81 个候选在 HR Poisson 下全部没有部署资格。fixed-size grouping 在 burst 可达到 0% singleton、99.38% batch utilization，但在 Poisson 只有 60.51%–83.18% singleton、27.04%–33.79% utilization。原因是调度器只对已经到达的 pending query 成组，又在 pending 非空时立即派发。

本实验不重新扫描 theta 或用 Finance 修配置，而是检验一个单独机制：**允许请求为局部性成组等待有限预算，并允许相邻 Jaccard group 共同填满物理 request batch；等待收益必须由 arrival-to-publication sojourn 支付。**

## 固定 grouping family

除两个等待参数外全部冻结为上一分支 HR score 最低但未部署的 fixed-Jaccard adaptation：

- Top-20 BM25 access set，W64；
- fixed-size greedy Jaccard agglomerative；
- target group size 16；
- group pool 64；
- active LRU capacity 80 pages；
- physical request batch max 8 queries；
- exact next-physical-batch-first-query prefetch；
- group execution order不变，但允许相邻 group 跨边界填满同一个 physical batch。

不再搜索 theta、target size、pool、capacity 或 prefetch 规则。fixed groups 只决定局部性执行次序；physical batch 是执行与发布原子单位。

## 有限等待与 queue trigger

候选网格只有 12 项：

- `wait_budget ∈ {0, 4, 16, 64}` unit-time；
- `min_pending ∈ {4, 8, 16}` queries。

当没有已冻结执行 plan、pending 非空且 pending 数小于 `min_pending` 时：

1. 以 pending 中最老 query 的 arrival time 加 `wait_budget` 为硬 deadline；
2. 只等待到“下一条 query 到达”或“deadline”两者较早者；
3. 一旦 pending 达到 `min_pending` 或 deadline 到达，立即对当前可见 pending 建组并派发；
4. 流结束时不得等待不存在的未来 query；所有 query 必须恰好发布一次。

`wait_budget=0` 是同一执行器的 no-wait anchor。没有候选可无限等待，且不得在看到 HR/Finance 结果后扩大网格。

## 主时间与成本定义

到达时间和服务共用一个 unit-time clock：

- build=1；reload=1；prefetch build/reload=1；hit=0；
- 无工作而等待下一次到达，以及 bounded grouping 主动等待，都会推进 elapsed unit-time，但不记为 build/reload work；
- query 的 `sojourn_unit_time = publication_clock - arrival_time`；
- 一个 physical batch 完成全部 demand 后原子发布，其中所有 query 使用同一 publication clock；
- `unit_work_per_query = (demand builds + demand reloads + prefetch builds + prefetch reloads) / query_count`。

主延迟指标是 arrival-to-publication sojourn，包含 queue wait、主动 grouping wait 和 build/reload service。completion pages 只作为兼容诊断，不能替代 sojourn。选择与 Finance gate 都用 mean sojourn 和 charged unit work，不用累计 publication timestamp 冒充成本。

所有被比较策略（FIFO、current overlap、history/static popularity、faithful、frontier、bounded CaGR）使用相同 unit-time arrival/service semantics、capacity=80 和 batch max=8。

## Cross-group fill 与必报审计

Jaccard group 按确定性顺序扁平化后，每 8 条组成 physical batch，因此一个 batch 可以跨 group boundary。每次回放必须报告：

- group singleton、mean/max size；
- batch utilization；
- batch group purity：每个 batch 中占比最大的 logical group 成员数 / batch size；
- cross-group batch fraction；
- prefetch build/reload/useful/wasted/unused unit work；
- demand build/reload/hit；
- mean/P50/P95/max sojourn；
- unit work per query；
- starvation fraction/max younger bypass；
- final union 与 dispatch/order hashes。

prefetch 不能免费。预取页若在首次 demand 前被淘汰，或流结束仍未使用，计 wasted；其 build/reload 已永久计入 unit work。

## 信息边界与固定 workload

- HR 开发域，Finance 封存域；K=20，burst 与 Poisson；
- permutation seeds 固定为 `20260804`–`20260808`；
- HR 选择只允许 BM25 manifest/runtime、access sets、arrival、cache/build/reload/prefetch 与上述延迟统计；
- 选择前禁止打开 HR qrel、HR visual、Finance 任意结果；
- 配置摘要冻结后，才加载 HR qrel/visual 做 post-hoc quality，并一次性打开 Finance；
- 不使用 GPU；成本是抽象 unit-time/unit-work，不是墙钟或能耗。

## HR 部署资格与选择准则

对每个候选聚合 HR burst/Poisson × 五 seeds，并与相同 arrival/service semantics 的 FIFO 比较。

部署资格同时要求：

1. `wait_budget` 属于冻结有限网格；
2. final union parity，query 无丢失、无重复；
3. burst 与 Poisson 各自 `P95 sojourn <= 1.05 × FIFO P95 sojourn`；
4. burst 与 Poisson 各自 starvation fraction `<=5%`。

不再用 singleton 或 batch utilization 作硬门；它们是机制解释指标。

令每个 arrival model 下：

- `R_sojourn = candidate mean sojourn / FIFO mean sojourn`；
- `R_work = candidate unit work per query / FIFO unit work per query`。

唯一选择分数：

`score = mean_over_{burst,poisson}(0.5 * R_sojourn + 0.5 * R_work)`。

只在合格候选中取最小 score。tie-break 依次为：较小 worst-case `max(R_sojourn,R_work)`、较小 mean P95-sojourn ratio、较小 wait budget、较小 min pending、配置 JSON 字典序。无合格候选则冻结为 NO-DEPLOYABLE，不从 Finance 反选。

## Finance 强判停门

HR 配置冻结后，在 Finance burst 与 Poisson 分别比较 bounded CaGR 与同 capacity frontier：

- `frontier_sojourn_adv = 1 - frontier_mean_sojourn / cagr_mean_sojourn`；
- `frontier_work_adv = 1 - frontier_unit_work_per_query / cagr_unit_work_per_query`。

只要任一 arrival setting 出现以下任一情况，结论为 **DOWNGRADE/STOP stronger-baseline claim**：

1. `frontier_sojourn_adv < 5%`；
2. `frontier_work_adv < 5%`；
3. bounded CaGR 在 mean sojourn 与 unit work 上均不劣于 frontier，且至少一项严格更好。

若 HR 无可部署候选，也直接 STOP，但必须写成“开发域部署资格不可证实”，不能伪装成 Finance counterexample。

只有唯一冻结的 bounded CaGR 在 Finance 两种到达下都让 frontier 的 sojourn 与 unit-work 优势至少 5%，且没有被 bounded CaGR Pareto dominance，才可写 **STRONGER BASELINE SURVIVED**。P95、quality regret、starvation 若明显恶化，仍需限制论文结论。

## 固定产物

- 本预注册合同；
- bounded-wait replay 与单元测试；
- HR 12 候选完整表、唯一冻结配置和 SHA-256；
- Finance sealed gate；
- 含全部输入/顺序/配置 hashes 的 JSON；
- 如实区分不可证实、counterexample 与 survived 的 Markdown 报告；
- 全仓测试与干净本地提交。
