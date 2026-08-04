# Hard-fair oracle headroom 预注册

日期：2026-08-05

分支：`exp/hard-fair-oracle`

基线提交：`ad11bbd`

状态：在实现、HR 扫描或 Finance 开封前冻结。

## 研究问题与范围

上一轮 60 点 soft-weight oracle family 中，24 个配置同时改善三个主轴并通过 P95，
但 starvation 为约 11%--23%；仅有的 4 个 starvation-safe 点退化为 FIFO。两个集合
完全不相交。本实验不再调连续权重，而是固定最强松弛效用 `oracle_15`，加入真正的
younger-bypass 硬约束，检验强收益能否在显式公平预算下保留。

这仍是允许看未来 arrival 和精确成本的 constrained oracle probe，不是 deployable
方法、数学最优解或全局 Pareto frontier。结果只能说明有限 hard-budget family 是否存在
headroom。

## 固定 workload、资源与端点

- HR 开发域、Finance 封存域；BM25 Top-20；burst/Poisson；seeds
  `20260804`--`20260808`；W64；
- active LRU 80 pages；physical batch max 8；atomic publication；
- unit clock：demand build/reload=1，hit=0；idle/explicit future wait 推进 elapsed，
  不增加 charged work；
- oracle 不做 prefetch，不服务尚未到达 query，不改变 candidate union；
- endpoints 固定为 FIFO、faithful CaGR theta=0.5、frontier、HR-selected bounded
  CaGR；相同 arrival、cache、demand 与 publication 语义；
- CPU only，不修改 endpoint 调度器。

## 固定效用：oracle_15

当硬约束允许多个候选时，完全沿用上一轮 relaxed minimax 最好的 `oracle_15`：

- `lambda_quality=0`；
- `lambda_completion=0.75`；
- `lambda_deadline=0.25`；
- deadline scale 256 unit-time；
- future wait budget 16 unit-time；
- cost denominator epsilon 1；
- 每个 batch 槽在 virtual compiled/LRU 上重算 exact next-query demand cost；
- utility tie-break 仍为更老 arrival rank、query id。

不扫描 lambda、deadline scale、wait、prefetch 或 qrel gain。本实验只测试硬公平约束的
独立作用。

## Younger-bypass 硬约束

对每个已到达但尚未 dispatch 的 query `j`，在线维护：

`bypass_j = 已经 dispatch 且 arrival rank 比 j 更年轻的 query 数`。

在 batch 的每个 slot，先按固定 `oracle_15` utility 找 unconstrained best。候选 `i`
只有在以下条件成立时才可选：对每个仍 pending 且比 `i` 更老的 query `j`，选择 `i`
后都有 `bypass_j + 1 <= B`。选择后立即更新这些更老 query 的 bypass count，再选择下一
slot。最老 pending 永远可行，因此约束不会死锁。

固定候选 budget 只有：

`B ∈ {8, 16, 32, 64}`。

不加入 absolute-age cap：本轮 starvation 的冻结定义本来就是 younger bypass，直接硬约束
该量最可审计；额外 age grid 会混入第二种安全定义。`B<64` 理论上保证冻结的
`bypass>=64` starvation 为 0；`B=64` 允许恰好 64，仍必须实测 fraction<=5%。

## 强制触发审计

每个 replay 必须报告：

- batch-slot selection 总数；
- unconstrained best 不可行、改选其他 query 的 forced selection 次数与比例；
- 被硬约束保护过的 unique query 数与比例；
- 最大在线/最终 younger bypass；
- configured budget 与实际 budget violation count（必须为 0）；
- future wait events/time；
- dispatch/order/trace hashes 与 final union parity。

## 三轴与系统指标

沿用 `a39e7ca` 的 population anytime quality，严格分开：elapsed unit time、charged unit
work、unique compiled pages。每个 domain×arrival×seed×axis 的共同 horizon 取四个
hard oracle candidate 和四个 endpoint 的最大终点；Finance 只含 HR 冻结的一个配置与
四个 endpoint。

报告三轴 AUC/regret、sustained T50/T90、固定预算质量，以及 mean/P95 sojourn、
work/query、starvation、cache/build/reload、约束触发率。

## HR 资格、Pareto 与 knee 选择

每个 hard budget 在 HR burst 与 Poisson 都必须满足：

1. union/query parity，hard-budget violation=0；
2. starvation fraction（final younger bypass>=64）`<=5%`；
3. `P95 sojourn <=1.05 * min(P95_bounded, P95_frontier)`；
4. mean sojourn `<= bounded`；
5. charged work/query `<= bounded`；
6. elapsed normalized quality regret `<= frontier`。

对合格点构造七维最小化向量：burst/Poisson 各三个主 ratio，加
`actual_max_younger_bypass/64`。先删除被另一合格点七维弱支配且至少一维严格支配的点。
在剩余 Pareto 点中，对每个维度用该集合的 observed min/max 做 `[0,1]` min-max；常数
维记 0。唯一 knee 是到 coordinate-wise ideal 原点的等权 Euclidean distance 最小点。
tie-break 依次为较小六主 ratio 的 maximum、较小六项平均、较小 configured B、配置
JSON 字典序。

若 HR 无合格点，冻结 `NO-QUALIFIED-HARD-ORACLE`，Finance gate 自动失败，不从 Finance
反选。否则冻结唯一配置与 SHA-256 后，才加载 Finance。

## Finance GO 门

同一个 HR-frozen hard oracle 必须在 Finance burst 与 Poisson 分别满足：

- parity、hard-budget violation=0；
- starvation<=5%；
- P95<=1.05×best endpoint；
- mean sojourn<=bounded；
- work/query<=bounded；
- elapsed quality regret<=frontier；
- 三个主轴中该 arrival 至少一项相对对应 endpoint 改善>=5%。

两种 arrival 全部通过才写 `HARD-FAIR HEADROOM GO`；否则写
`NO HEADROOM IN REGISTERED HARD-FAIR FAMILY`。即使 GO，也只能证明 finite constrained
oracle headroom，不能称为 deployable scheduler；oracle 的未来信息和 exact cost 仍需
蒸馏为在线 proxy。

## 固定产物

- 本合同与独立提交；
- hard feasibility、slot update、no-deadlock、trigger accounting 和 gate 的小例测试；
- HR 四点完整表、Pareto/knee、冻结 SHA，Finance 一次性结果；
- 三轴/系统/约束审计 JSON 与 Markdown 报告；
- 全仓测试、deterministic JSON 与 clean local commit，不推远程。
