# Multi-objective oracle headroom 预注册

日期：2026-08-04

分支：`exp/multiobjective-oracle-headroom`

状态：在实现 oracle、扫描 HR 或打开 Finance 前冻结。

## 研究问题与结论边界

本实验不是提出 deployable scheduler，而是探测已观测 tradeoff 的可达空间：如果一个
允许看未来 arrival、冻结 qrel quality gain 和精确当前成本的强 oracle 仍不能同时达到
系统效率与时间质量端点，则下一步应优先改 measurement/representation；如果存在
headroom，再尝试把 oracle 决策蒸馏成无 qrel proxy。

注册 family 只是有限 greedy oracle probe，不是整数规划最优解、全局 Pareto frontier
或“不可能性证明”。只能写 `HEADROOM GO` 或
`NO HEADROOM IN REGISTERED FAMILY`，不得把 family 内最好点称为 mathematical oracle
optimum。

## 固定 workload、资源与端点

- HR 开发域、Finance 封存域；BM25 Top-20；burst/Poisson；seeds
  `20260804`--`20260808`；W64；
- active LRU 80 pages；physical batch max 8；atomic batch publication；
- 同一 unit clock：demand/prefetch build 或 reload=1，hit=0；idle 与 explicit wait
  推进 elapsed 但不增加 charged work；
- frozen endpoints：FIFO、faithful CaGR theta=0.5、current frontier、HR-selected
  bounded CaGR（fixed Jaccard 16、pool64、wait16、min-pending4）；
- oracle 不做 prefetch。它与 endpoints 使用相同 demand、persistent compiled set、LRU、
  arrival release、batch publication 和成本。该选择避免额外设计机制混入 headroom probe；
- CPU only，不修改任何 endpoint 调度器。

## Oracle 的允许信息

Oracle 在每次 batch 决策时允许观察：

- 全部 query 的未来 arrival timestamp 和 arrival rank；
- 每个 query 的冻结 post-hoc fusion gain
  `g_i = fusion_nDCG@10_i - BM25_nDCG@10_i`；
- 当前 compiled set、active LRU 顺序，以及候选 query 若下一条执行时的精确 demand
  build/reload/hit 成本；
- pending query 的当前 age。

Oracle 不允许提前发布尚未到达的 query，也不改变页面候选集合、质量定义或最终 union。
这些信息特权必须在论文中写成 upper-bound probe，不能伪装成可部署方法。

## 冻结 greedy batch rule

每次从最老的 W64 pending prefix 逐槽填充最多 8 个 query。每选一条 query 后，在
virtual compiled/LRU 状态中按页面排序精确模拟其 demand，再重算下一槽所有可见候选的
边际成本。对候选 `i`：

- `c_i`：在当前 virtual state 下一条执行的 charged demand cost；hit-only query
  令分母为 `max(c_i,1)`；
- quality-density：`(g_i / max(c_i,1)) / max_j abs(g_j/max(c_j,1))`，分母为零时
  全部记 0，保留负 gain；
- completion-density：`(1/max(c_i,1)) / max_j(1/max(c_j,1))`；
- deadline pressure：`min(max(now-arrival_i,0)/D, 1)`。

唯一 score：

`lambda_q * quality_density + lambda_c * completion_density + lambda_d * deadline_pressure`。

取最大 score；tie-break 为更老 arrival rank，再按 query id。batch 内成本模拟与真实
demand 使用相同 sorted-page/LRU 语义。quality gain 只参与 score；实际总体质量仍在
整个 physical batch 完成后原子更新。

## Future-arrival wait

若没有冻结 batch plan、pending 非空但不足 8 条，oracle 可使用 `future_wait_budget`：

1. deadline 为当前最老 pending arrival 加 wait budget；
2. 等到下一 query arrival 或 deadline 的较早者；
3. pending 达到 8、deadline 到达或没有未来 query 时立即派发；
4. wait 完整进入 elapsed/sojourn，不进入 charged work。

这是 oracle 使用未来 arrival 的唯一主动等待机制；尚未到达 query 永远不能进入 batch。

## 固定候选 family

- `(lambda_q, lambda_c, lambda_d)` 扫描非负、和为 1、步长 0.25 的完整 simplex：
  15 个固定权重点；
- deadline scale `D ∈ {64, 256}` unit-time；
- `future_wait_budget ∈ {0, 16}` unit-time；
- cost denominator epsilon 固定为 1；
- 总计 60 个配置。不得在看到 HR/Finance 后扩 grid、改归一化、加入 beam 或改 tie-break。

该 family 同时包含 quality-only、completion/locality-only、deadline-only 和混合规则。
它足以回答固定 greedy class 的 headroom，但不代表所有 clairvoyant scheduling。

## 三轴质量与系统指标

沿用 `a39e7ca` 的 post-hoc population quality：未发布 query 使用 BM25 quality，已发布
query 使用 frozen fusion quality。每个 atomic publication 同时记录：

- elapsed unit time；
- charged unit work；
- unique compiled pages。

在 HR 每个 domain×arrival×seed×axis，统一 horizon 取 60 个 oracle candidate 和四个
endpoint 的最大终点；Finance 只打开 HR 冻结的一个 oracle 配置，horizon 取该配置和四个
endpoint 的最大终点。报告三轴 AUC/normalized regret、sustained T50/T90 和固定预算
质量，并报告 mean/P95 sojourn、work/query、starvation、union、order/trace hashes。

## HR 选择与 Finance 封存

HR 本轮明确允许 oracle 使用 HR qrel/frozen gain。先跑 60 个配置。部署安全资格只要求
burst 与 Poisson 均满足：

- final union/query parity；
- `P95 sojourn <= 1.05 * min(P95_bounded, P95_frontier)`；
- starvation fraction（younger bypass >=64）`<=5%`。

对每个安全配置、每种 arrival 定义三个 ratio：

- mean sojourn / bounded mean sojourn；
- work/query / bounded work/query；
- elapsed normalized quality regret / frontier regret。

唯一选择规则是最小化六个 ratio 的 maximum；tie-break 依次为六项平均、最大 P95 ratio、
最大 starvation、配置 JSON 字典序。配置 JSON 与 SHA-256 冻结后才加载 Finance。若 HR
没有安全配置，则冻结 `NO-SAFE-ORACLE`，Finance gate 自动失败，不从 Finance 反选。

## Finance Headroom GO 门

同一个 HR-frozen oracle 配置必须在 Finance burst 与 Poisson **分别同时**满足：

1. mean sojourn `<= bounded CaGR`；
2. charged work/query `<= bounded CaGR`；
3. elapsed normalized quality regret `<= frontier`；
4. P95 sojourn `<=1.05 * min(P95_bounded, P95_frontier)`；
5. starvation `<=5%`；
6. 在前三个主轴中，该 arrival 至少一项相对对应端点改善 `>=5%`。

两种 arrival 全部通过才写 `HEADROOM GO`。否则写
`NO HEADROOM IN REGISTERED FAMILY` 并转向 measurement/representation；这不表示全局
无解，也不能隐去最接近门的 ratio 和失败约束。

## 固定产物

- 本预注册合同与独立提交；
- oracle replay、精确 virtual cost/LRU、future wait 和 gate 的可手算测试；
- HR 60 点完整表、冻结配置/hash、Finance 一次性结果；
- 含三轴指标、系统/安全指标、输入/order/trace hashes 的 JSON；
- 如实判定 headroom 的 Markdown 报告；
- 全仓测试、确定性复跑和干净本地提交。
