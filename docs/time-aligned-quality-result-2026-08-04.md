# Time-aligned anytime quality 结果

日期：2026-08-04  
分支：`exp/time-aligned-quality`  
预注册：`a39e7ca`  
机器：CPU only，未使用 GPU  
机器可读结果：`results/systems/time-aligned-quality-v1.json`

## 一句话结论

Finance 仍应写成 **系统效率—质量进度 tradeoff**，而不是 bounded CaGR 的全局支配。
在 burst 下 frontier 的 elapsed-time quality regret 比 bounded CaGR 低 2.66%；在
Poisson 下结论反转，bounded CaGR 低 2.55%。因此 bounded CaGR 的 sojourn/work
反例有效，但没有跨两种到达模型稳定支配时间对齐质量。

同时，新指标改变了旧 unique-page 结果的直觉强度：frontier 每个新页面带来的质量顺序
明显更好，但 CaGR 更少 reload、完成整个查询流更早；把真实服务进度放回横轴后，两种
效应相互抵消并产生曲线交叉，而不是某个调度器全程领先。

## 计量语义

每个 query 未发布时使用 BM25 nDCG@10，所在 physical batch 完成全部 demand 后，
原子切换为冻结的 Top-20 z-score fusion nDCG@10。总体质量是所有 query 当前可见
质量的均值。质量只在发布点改变，允许下降，不做裁剪或单调化。

三条横轴严格分开：

- `elapsed_unit_time`：build、reload、prefetch 各为 1，hit 为 0；包含 idle 和
  explicit bounded wait；
- `charged_unit_work`：只包含 demand/prefetch build/reload；
- `unique_compiled_pages`：只包含首次构建的不同页面，即旧 normalized regret 的真实
  横轴。

prefetch 在当前 batch 发布之后执行，因此不延迟当前质量切换，但会推迟下一次发布。
每个 domain×arrival×seed×axis 使用全部方法最大终点作为共同 horizon；早完成的方法
保持最终质量到共同 horizon。固定预算因此是真正相同的 clock/work/page 预算。

## Finance 主结果

遗憾越低越好。T50/T90 是持续达到最终增益 50%/90% 的共同 clock 比例，不是第一次
短暂越线。

| arrival | 方法 | elapsed regret↓ | charged-work regret↓ | unique-pages regret↓ | sustained T50↓ | sustained T90↓ | 25% clock质量↑ | 50% clock质量↑ | 完成clock比例↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | bounded CaGR | 0.4190 | 0.4190 | 0.6538 | 0.408 | 0.743 | 0.5379 | 0.5491 | 0.840 |
| burst | frontier | **0.4078** | **0.4078** | **0.5026** | 0.428 | 0.844 | **0.5398** | **0.5509** | 0.860 |
| Poisson | bounded CaGR | **0.4123** | **0.4105** | 0.6387 | 0.459 | **0.738** | 0.5379 | 0.5482 | **0.839** |
| Poisson | frontier | 0.4231 | 0.4231 | **0.5132** | **0.440** | 0.804 | **0.5391** | **0.5497** | 0.875 |

burst 中没有发生主动等待，所以 selected bounded 与 no-wait anchor 完全一致。frontier
早期固定预算质量更高，因而积分遗憾更低；但 bounded 更早完成，持续 T50/T90 反而更早。
这说明两条质量曲线发生交叉，单报 T90 同样会误导。

Poisson 中 bounded 的 elapsed regret 为 0.4123，frontier 为 0.4231，按预注册三个
主轴可称该 arrival setting 下的三轴 dominance；但这不是 pointwise dominance：
frontier 的 25%/50% clock 质量与 T50 仍略好，bounded 在后半程赶超，并更早持续达到
T90 和完成全流。论文必须保留这层限制。

## 两个系统轴没有改变

新增 publication trace 没有改变调度或旧系统统计。完整重跑的全部 HR/Finance、
burst/Poisson、八个方法 aggregate 与 bounded reference 精确相同。

| Finance | bounded mean sojourn | frontier | bounded work/query | frontier |
|---|---:|---:|---:|---:|
| burst | **2,204.44** | 2,302.32 | **15.829** | 16.199 |
| Poisson | **1,886.81** | 2,041.67 | **15.762** | 16.477 |

所以本实验没有推翻 Finance 系统轴反例；它只修正了第三条质量轴的解释。

## 为什么三条质量轴给出不同答案

unique-pages regret 只问“每构建一个从未见过的页面，质量推进多快”。frontier 在该轴上
稳定领先，Finance burst/Poisson 分别为 0.5026/0.5132，bounded 为
0.6538/0.6387。它证明 frontier 的新页面选择更有质量价值。

elapsed 与 charged-work 还会支付 reload 和 prefetch。bounded 的局部性让它少做大量
reload，并更早完成所有 query；完成后最终质量在共同 horizon 上保持。因此 frontier 的
新页面优势不再自动等于更好的在线质量曲线。旧报告将 unique pages 简写成
“quality-work”掩盖了这个差别。

大多数 burst replay 从 t=0 起持续有 backlog，所以 elapsed 与 charged work 相同。
Poisson selected bounded 包含五次显式等待，Finance 平均 charged-work regret 为
0.4105，而 elapsed regret 为 0.4123；这个差值正是等待时间的真实代价，未被隐藏。

## 兼容方法检查

Finance elapsed regret 排名如下：

| 方法 | burst↓ | Poisson↓ |
|---|---:|---:|
| frontier | **0.4078** | 0.4231 |
| current overlap | 0.4088 | 0.4410 |
| bounded CaGR / no-wait | 0.4190 | 0.4213（no-wait） |
| HR-selected bounded CaGR | 0.4190 | **0.4123** |
| offline static popularity | 0.4698 | 0.4641 |
| history popularity | 0.4836 | 0.4682 |
| faithful CaGR theta=0.5 | 0.5049 | 0.5056 |
| FIFO | 0.5146 | 0.5159 |

这也说明结果不是“所有 grouping 都更好”：faithful theta=0.5 接近 FIFO；优势来自冻结的
fixed-Jaccard deployment bundle，而不是 CaGR 名称本身。

## HR post-hoc 检查

HR 上 selected bounded 的 elapsed regret 为 0.3707（burst）和 0.3744
（Poisson），frontier 为 0.4327 和 0.4292。该结果只作开发域解释，不能用于重新选择
配置；配置早已由 access-only 系统指标冻结。HR qrel 和 visual 仍只在 post-hoc 阶段
打开。

## 论文应该怎么写

当前数据支持：

1. frontier 优化 unique-new-page quality efficiency；
2. fixed-Jaccard bounded CaGR 优化复用、reload 与完成时间；
3. 真正的在线质量由“新证据价值”和“物理复用成本”共同决定，单用 unique pages 或
   单用 unit work 都不能代表部署时间；
4. Finance burst 是清晰 tradeoff，Poisson 在三个预注册平均主轴上 bounded 占优，
   但曲线仍交叉。

不能写：

- bounded CaGR 在所有到达模型全局支配 frontier；
- frontier 的 unique-page regret 优势等于真实时间质量优势；
- sustained T90 或任一固定预算单点足以代表整条 anytime 曲线；
- 抽象 unit clock 是 GPU 墙钟、能耗或答案质量。

下一步方法设计应直接优化 time-aligned quality/locality frontier：在 scheduler score 中
同时估计新页面的质量增益与 reload/cache 代价，并用无 qrel 的 proxy 学习这个权衡。
继续单独调 wait 的价值有限。

## 审计与复现

- 所有 quality replay 的 dispatch order 与 zero-gain replay 一致；qrel 不进入调度；
- 所有 system aggregate 与 `cagr-bounded-wait-v1.json` 精确一致；
- JSON 保存每个 publication trace hash、每 seed 的三轴共同 horizon、完整 per-run
  metric、输入 provenance 和冻结配置 hash；
- bounded reference SHA-256：
  `0a57cca8755ed9d187bf95953471ad10fb31cea26feec791c221af2872efd151`；
- time-aligned JSON SHA-256：
  `ea618d84cc9306a9734841003f28f2a966cd8212938744ad69bcde06d8670017`。

复现命令：

```bash
PYTHONPATH=. python tools/analyze_time_aligned_quality.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --bounded-reference results/systems/cagr-bounded-wait-v1.json \
  --output results/systems/time-aligned-quality-v1.json
```
