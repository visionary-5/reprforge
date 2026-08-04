# Multi-objective oracle headroom 结果

日期：2026-08-04

分支：`exp/multiobjective-oracle-headroom`

预注册：`69add9b`

机器：CPU only，未使用 GPU

机器可读结果：`results/systems/multiobjective-oracle-headroom-v1.json`

## 一句话结论

最终判定是 **NO HEADROOM IN REGISTERED FAMILY**，但原因不是三个主目标没有可达
空间，而是注册的连续加权 greedy family 无法同时满足硬 starvation 约束。

HR 60 个 oracle 配置中：

- 24 个同时在 mean sojourn、work/query、elapsed quality regret 三个主轴优于对应
  endpoint，并通过 P95 门；
- 4 个通过 starvation≤5%；
- 两个集合完全不相交，因此安全候选为 0，不能冻结配置，也不得从 Finance 反选。

这只能说明注册的 60-config greedy oracle family 没有合格 headroom，不能写成全局
不可能、数学最优失败或表示层没有 headroom。

## Oracle family 与信息特权

Oracle 允许查看未来 arrival、冻结 qrel fusion gain、当前 compiled/LRU 状态和精确
下一查询 demand 成本。每个 batch 槽在 virtual LRU 上重算：

- signed marginal quality gain / charged cost；
- completion gain / charged cost；
- deadline pressure。

扫描步长 0.25 的三权重 simplex、deadline scale 64/256、future wait 0/16，共
60 个固定配置。它没有 prefetch，不能提前服务未到达 query。该 family 是有限 greedy
upper-bound probe，不是全局 oracle optimum。

## 最接近 GO 的松弛点

忽略 starvation 安全门时，HR minimax 最好的配置是 `oracle_15`：

- quality weight 0；completion weight 0.75；deadline weight 0.25；
- deadline scale 256；future wait 16；
- config SHA-256：
  `aaba4ca559ef966d4eaa0432b720dba4eb4f028145459e2fdcdcff40a8b471d0`。

下表是相对 endpoint 的 ratio，低于 1 越好。sojourn/work 的 endpoint 是 bounded
CaGR，elapsed regret 的 endpoint 是 frontier，P95 endpoint 取二者较小值。

| HR arrival | mean sojourn ratio↓ | work/query ratio↓ | elapsed regret ratio↓ | P95 ratio↓ | starvation |
|---|---:|---:|---:|---:|---:|
| burst | **0.7849** | **0.8214** | **0.7179** | 0.8156 | **12.26%** |
| Poisson | **0.7726** | **0.8282** | **0.7264** | 0.8185 | **12.26%** |

也就是：

- mean sojourn 改善 21.5% / 22.7%；
- charged work/query 改善 17.9% / 17.2%；
- elapsed quality regret 改善 28.2% / 27.4%；
- P95 显著低于安全上限；
- 但 younger-bypass≥64 的 query 占 12.26%，超过 5% 门约 2.45 倍。

绝对值也很直接：

| HR arrival | 方法 | mean sojourn | P95 sojourn | work/query | elapsed regret |
|---|---|---:|---:|---:|---:|
| burst | oracle_15 | **1,544.98** | **2,989.2** | **11.541** | **0.3101** |
| burst | bounded / frontier endpoint | 1,968.32 / 2,205.51 | 3,665 / 4,400.6 | 14.050 / 15.120 | — / 0.4320 |
| Poisson | oracle_15 | **1,289.18** | **2,580.8** | **11.685** | **0.3102** |
| Poisson | bounded / frontier endpoint | 1,668.61 / 1,958.85 | 3,153.3 / 4,335.2 | 14.109 / 15.598 | — / 0.4271 |

所以“质量与局部性不能同时改善”已经被 HR oracle envelope 否定；真正未解决的是在这些
收益下保护最老 query。

## 为什么 4 个 starvation-safe 点很差

仅有的四个 starvation-safe 配置全部是 `lambda_deadline=1`、其他权重为 0，只在
deadline scale 64/256 与 future wait 0/16 上不同。deadline pressure 对 age 单调，
再加更老 arrival 的 tie-break，这四个配置实质恢复 FIFO 顺序。

不等待的 deadline-only 点在两种 HR arrival 下分别约为：

| 指标 ratio | burst | Poisson |
|---|---:|---:|
| mean sojourn / bounded | 1.337 | 1.379 |
| work/query / bounded | 1.271 | 1.266 |
| elapsed regret / frontier | 1.140 | 1.151 |
| P95 / best endpoint | 1.332 | 1.358 |
| starvation | 0% | 0% |

因此其 minimax ratio 约 1.38，并且 P95 远超 1.05 门。future wait=16 只会让 Poisson
延迟进一步恶化。软 deadline 权重一旦混入 completion 或 quality，主轴/P95 会迅速
改善，但 starvation 跳到至少约 11%，没有形成安全插值区间。

## qrel-aware 点说明了什么

最强 minimax 点 `oracle_15` 的 quality 权重为 0；这说明大量 headroom 首先来自精确
completion-cost/locality，而不是 qrel 特权。

最好的 qrel-aware minimax 点为 quality=0.25、completion=0.75、deadline=0，
burst/Poisson 的 elapsed regret 仅为 frontier 的 0.427/0.461，sojourn 和 work 也都
改善约 11%–15%；但 starvation 升至约 17%。qrel 信息确实能显著推动早期质量，却会
进一步放大公平性问题。

这也是为什么本轮不能把“oracle 看了 qrel”包装成方法贡献。可部署方法仍需用无 qrel
proxy，并显式约束尾部。

## Finance 为什么没有 oracle 结果

预注册要求先在 HR 安全候选中唯一选择，再冻结配置 SHA 后一次性运行 Finance。由于
HR `safe_candidate_count=0`：

- selection 冻结为 `NO-SAFE-ORACLE`；
- null selection SHA-256 为
  `74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b`；
- Finance 不运行任何 oracle config，gate 自动判
  `NO HEADROOM IN REGISTERED FAMILY`；
- Finance 只保留 endpoints 和输入 provenance，不能从 Finance 挑 `oracle_15` 或其他
  看起来好的配置。

所以本轮没有 Finance 上“差一点通过”的 oracle 数字，也不应虚构跨域 transfer 结论。

## 对下一步方向的判定

按预注册 gate，本分支停止继续扫 lambda。最有价值的下一步是 measurement/fairness
审计，而不是再扩大 soft-weight grid：

1. 同时报 younger-bypass、绝对等待时间和 slowdown，确认 64 次越序是否准确代表用户
   可感知 starvation；
2. 构建带硬 age/bypass budget 的 constrained oracle，检验 12.26% 违规能否以小代价
   修复；该实验必须另行预注册，不能补进本轮 60 点；
3. 若硬约束后 envelope 仍消失，再把主力转向 representation/measurement；若保留，
   则蒸馏 completion-cost 与 quality proxy。

本轮数据更像是“软 Lagrangian 无法表达硬公平边界”，而不是“多目标方向没有空间”。

## 审计与复现

- 60/60 配置 final union/query parity；
- 24 个 primary-qualified 集合与 24 个 P95-qualified 集合完全相同；
- 4 个 starvation-qualified 点与上述集合交集为空；
- FIFO、faithful、frontier、bounded 四个 endpoint 的全部 system aggregate 和 order
  hash 与 `time-aligned-quality-v1.json` 精确一致；
- JSON 保存 60 点完整表、每 seed 三轴指标、dispatch/trace hash、future-wait、输入
  provenance 和封存审计；
- bounded reference SHA-256：
  `0a57cca8755ed9d187bf95953471ad10fb31cea26feec791c221af2872efd151`；
- time-aligned reference SHA-256：
  `ea618d84cc9306a9734841003f28f2a966cd8212938744ad69bcde06d8670017`；
- oracle result SHA-256：
  `e0d016c357add3d21fd66c0f59db417980999a7655e2e3eaefeee32fd0809230`。

复现命令：

```bash
PYTHONPATH=. python tools/analyze_multiobjective_oracle_headroom.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --bounded-reference results/systems/cagr-bounded-wait-v1.json \
  --time-reference results/systems/time-aligned-quality-v1.json \
  --output results/systems/multiobjective-oracle-headroom-v1.json
```
