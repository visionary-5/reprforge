# Time-aligned anytime quality 预注册

日期：2026-08-04

分支：`exp/time-aligned-quality`

状态：在修改 replay、加载 qrel 或生成结果前冻结。

## 研究问题

bounded CaGR 在 Finance 的 mean arrival-to-publication sojourn 与 charged
unit work/query 上反向支配 current frontier，但旧的 `quality_work_auc` 横轴实际是
unique compiled pages，不是时间，也不包含 reload、prefetch 和主动等待。本实验只修正
计量，不改变任何调度器，回答：在同一个 elapsed unit-time 时钟上，Finance 结果仍是
“系统效率换质量进度”，还是 bounded CaGR 连时间对齐质量也不劣于 frontier。

## 冻结输入与方法

- HR、Finance，BM25 Top-20 access set；burst 与 Poisson；seeds
  `20260804`--`20260808`；W64、active LRU 80、physical batch 8；
- 沿用已冻结的 HR-selected bounded CaGR：fixed Jaccard、target 16、pool 64、
  wait budget 16、min pending 4、cross-group fill；
- 同时报 FIFO、frontier、faithful CaGR theta=0.5、bounded no-wait anchor，以及现有
  replay 可直接兼容的 current-overlap、history-popularity、offline static-popularity；
- 不搜索参数、不修改 dispatch、group、prefetch、cache 或 arrival 逻辑，不用 GPU；
- 每个方法必须与上一结果的 dispatch order hash、system metrics 和最终 union 一致。

## 冻结质量状态

对 query `i`，`b_i` 是 BM25 nDCG@10，`r_i` 是冻结的 Top-20 z-score fusion
nDCG@10。调度器只观察文本 access set；`b_i`、`r_i` 和 qrel 只在 batch 已经选定、
完成之后用于 post-hoc 计量。

初始时所有 query 使用 `b_i`。physical batch 完成全部 demand 后原子发布；该 batch
中的 query 同时从 `b_i` 切换为 `r_i`。时刻 `x` 的总体质量为：

`Q(x) = mean_i(r_i if query i has published by x else b_i)`。

质量允许下降或超过最终值，不做单调化或裁剪。相同坐标上的多个零成本发布按执行顺序
发生；在该坐标预算上观察最后一个已发布状态。

## 三条严格分离的横轴

每个 atomic publication 记录同一个状态对应的三个坐标：

1. `elapsed_unit_time`：统一 arrival/service clock。build、reload、prefetch
   build/reload 各推进 1，hit 推进 0；等待下一次 arrival 和 explicit bounded wait
   也推进此轴。因此它包含 service、prefetch、idle 与主动等待。
2. `charged_unit_work`：只累计 demand/prefetch build/reload，每项 1；hit、idle、
   explicit wait 为 0。
3. `unique_compiled_pages`：只累计首次 build 的不同页面。它是旧
   `quality_work_auc` 的实际横轴，不得再简称为 elapsed 或 charged work。

prefetch 在当前 batch 发布后执行，所以不延迟当前 batch 的质量切换，但会推进
elapsed 与 charged-work 坐标，并延迟下一次 publication。所有曲线从坐标 0、全 BM25
质量开始；方法完成后，最终融合质量保持不变。

## 共同 horizon 与 anytime 指标

在每个 `(domain, arrival model, seed)` 比较单元内，对每条轴分别取所有方法终点的
最大值作为共同 horizon `H_x`。较早完成的方法在其终点至 `H_x` 保持最终质量。这样
固定预算是同一个绝对 clock/work/page 预算，不按各方法自己的完成时间偷偷缩放。

对每条轴 `x` 报告：

- step-function `mean_quality_auc = integral_0^H Q(x) dx / H`；
- 若最终增益 `Q_f-Q_0 > 0`，归一化增益
  `G(x)=(Q(x)-Q_0)/(Q_f-Q_0)`，以及
  `normalized_regret_auc = 1 - integral_0^H G(x) dx / H`；
- sustained T50/T90：最早的坐标，使从该坐标到共同 horizon 的全部质量状态持续满足
  `G>=0.5/0.9`；同时报告绝对坐标和 `coordinate/H_x`；
- 共同 horizon 的 10%、25%、50%、75%、100% 固定预算处，报告绝对质量、相对 BM25
  增益和最终增益完成比例。

若最终增益不为正，归一化 regret、T50/T90 和最终增益比例均为 null；绝对质量 AUC、
带符号增益 AUC和固定预算绝对质量仍报告。曲线非单调时 T50/T90 必须使用 suffix
minimum 的“持续达到”，不能使用第一次短暂越线。

主表首先报告 elapsed-time 指标；charged-work 和 unique-compiled-pages 使用完全相同
算法单列，字段名必须带 axis，三者不得混称。旧指标保留为兼容审计，并明确重命名为
`unique_compiled_pages` 语义。

## Finance 判读

对每种 arrival model 单独比较 HR-selected bounded CaGR 与 frontier：

- 若 bounded CaGR 保持 mean sojourn 与 charged work/query 两轴不劣，并且
  elapsed-time normalized quality regret 也不高于 frontier，则称为预注册三轴
  dominance；至少一轴必须严格更好；
- 若 bounded CaGR 仍在两个系统轴占优，但 frontier 的 elapsed-time normalized quality
  regret 更低，则结论是明确 tradeoff；
- sustained T90、25%/50% clock-budget quality、P95 sojourn 和 starvation 是强制报告的
 解释与安全指标，不用事后替换主判据。

不得把单个 arrival model 的 dominance 外推到另一个 model、HR、真实墙钟、GPU
吞吐、答案质量或未测试语料。

## 固定产物

- 本合同及其独立提交；
- replay publication trace、三轴 metric 实现与可手算单元测试；
- HR/Finance 全方法、五 seeds 的 JSON，含输入/配置/order hashes；
- Markdown 结果报告，明确回答 Finance 是 tradeoff 还是真实三轴 dominance；
- 全仓测试和干净本地提交。
