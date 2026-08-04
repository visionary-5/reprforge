# Causal Hard Frontier 结果

日期：2026-08-05

分支：`exp/causal-hard-frontier`

预注册：`f0ab00e`

机器：CPU only；没有下载数据、生成 embedding 或调用 GPU

机器可读结果：`results/systems/causal-hard-frontier-v1.json`

## 一句话结论

预注册判定是 **PAPER METHOD CANDIDATE; CROSS-RETRIEVER UNVERIFIED**。

上一轮 B32 constrained-oracle 的收益已经被固化为独立事件驱动 policy
`hard_budget_frontier`。它的构造器不接收策略参数，公开接口不能接收 qrel gain、完整
arrival order/time array、下一 arrival 或 end-of-stream。HR/Finance 的 20 个
domain×arrival×seed cell 与 B32 reference 在 dispatch、elapsed、work、cache、union、
bypass 和 publication trace 上全部精确等价。

同一冻结 policy 在 HR、Finance、Computer Science、Industrial、Pharmaceuticals 五个
ViDoRe v3 域全部完成 transfer。相对 bounded CaGR，跨 10 个 domain×arrival cell 的平均
mean sojourn 改善 7.23%，work/query 改善 4.71%；相对 frontier，elapsed quality regret
平均改善 6.87%。所有 cell P99 都在注册安全线内、B32 violation 为 0。

本机只有 ColModernVBERT 结果 summary 和远端 artifact 指针，没有可直接读取的原始
per-query replay trace；按合同没有复制或重算，因此不能声称跨 retriever 已验证。

## 从 oracle 到在线方法

`HardBudgetFrontier` 固定：B32、completion 0.75、deadline 0.25、deadline scale 256、
timeout 16、batch 8、W64、LRU 80。没有 transfer 调参。

方法只处理四类当前信息：

1. 单个已发生的 `arrival(query_id, locator_cohort, event_time)`；
2. 已设置 deadline 到期的 `timer(event_time)`；
3. 当前 compiled pages 和 active LRU；
4. 当前 pending 的 arrival rank/age 与在线 bypass counter。

未满 batch 时，它为最老 pending query 设置 16-unit timer；中间 arrival 只作为事件逐条
加入，不查看下一事件时间或内容。batch 每个 slot 都重新取 `pending[:64]`，因此一个 slot
释放后，第 65 个 pending query 可以进入窗口；显式 `pending>64` 测试覆盖了这个边界。

qrel/visual score 只在 batch 已选完并完成后由 evaluator 用来发布质量曲线。交换 gain 的
测试得到相同 dispatch/work/bypass；scheduler 对象状态中也不存在 quality、qrel 或 future
trace。

## 20-cell reference 等价硬门

| 域 | arrival | seeds | exact cells | dispatch | elapsed tuples | work/cache/union | bypass/trace |
|---|---|---:|---:|---|---|---|---|
| HR | burst | 5 | 5/5 | exact | exact | exact | exact |
| HR | Poisson | 5 | 5/5 | exact | exact | exact | exact |
| Finance | burst | 5 | 5/5 | exact | exact | exact | exact |
| Finance | Poisson | 5 | 5/5 | exact | exact | exact | exact |

这里的 elapsed exact 包括 per-query completion unit-time、sojourn unit-time 与每个 atomic
publication point。参考方法虽然函数参数中仍有 qrel gain，但其冻结 quality weight 为 0；
新方法用不含该参数的独立 policy 得到了 20/20 完全相同的行为，因而不再只是“oracle
observability 的解释”，而是实际代码边界上的 causal materialization。

## 五域规模与 parity

| ViDoRe v3 域 | queries | corpus pages | Top-20 union pages | 10 runs | parity | max bypass | B32 violation |
|---|---:|---:|---:|---:|---|---:|---:|
| HR | 318 | 1,110 | 895 | 10 | pass | 32 | 0 |
| Finance-EN | 309 | 2,942 | 1,855 | 10 | pass | 32 | 0 |
| Computer Science | 215 | 1,360 | 1,144 | 10 | pass | 32 | 0 |
| Industrial | 283 | 5,244 | 2,129 | 10 | pass | 32 | 0 |
| Pharmaceuticals | 364 | 2,313 | 1,797 | 10 | pass | 32 | 0 |

五域共 50 个 hard-policy replay。burst 不触发 timer；Poisson 每域 5 seeds 共触发 5 次
deadline timer。所有数据来自已有本地 NPZ，并逐文件核对 frozen domain-matrix SHA。

## 跨域主指标

下表为 hard method 的 ratio，越低越好。sojourn/work 对 bounded CaGR；elapsed regret 对
注册的 frontier reference；P99 对 bounded/frontier 中较低者。

| 域 | arrival | sojourn ratio↓ | work ratio↓ | elapsed regret ratio↓ | P99 ratio↓ | 任一轴改善≥5% |
|---|---|---:|---:|---:|---:|---|
| HR | burst | 0.932 | 0.950 | **0.824** | 0.949 | yes |
| HR | Poisson | 0.923 | 0.956 | **0.851** | 0.940 | yes |
| Finance | burst | **0.919** | **0.942** | 0.995 | 0.941 | yes |
| Finance | Poisson | **0.922** | **0.949** | 0.963 | 0.940 | yes |
| Computer Science | burst | **0.911** | **0.943** | **0.944** | 0.936 | yes |
| Computer Science | Poisson | **0.921** | **0.946** | **0.912** | 0.928 | yes |
| Industrial | burst | 0.953 | 0.975 | 0.987 | 0.974 | no |
| Industrial | Poisson | **0.949** | 0.971 | **0.953** | 0.968 | yes |
| Pharmaceuticals | burst | **0.923** | **0.947** | 0.974 | 0.953 | yes |
| Pharmaceuticals | Poisson | **0.923** | **0.949** | **0.912** | 0.943 | yes |

跨 cell median ratio 为 0.923/0.949/0.948，即 median 改善约 7.69%/5.08%/5.16%；9/10
cell 至少一个主轴改善 5%。Industrial burst 是唯一没到 5% 的 cell，但三个主 ratio 仍都
小于 1，未出现 transfer regression。10/10 P99 ratio 小于 1，强于注册的 80% 安全门。

## 与五个固定方法的总体比较

下表是 10 个 domain×arrival cell 的 arithmetic mean。sojourn/work 仍归一化到 bounded，
elapsed regret 归一化到 frontier。

| 方法 | mean sojourn ratio↓ | work ratio↓ | elapsed regret ratio↓ | mean P99 | mean unique-page regret↓ |
|---|---:|---:|---:|---:|---:|
| FIFO | 1.260 | 1.202 | 1.197 | 4,674 | 0.728 |
| frontier | 1.072 | 1.052 | 1.000 | 4,628 | **0.583** |
| overlap-only | 1.016 | 1.055 | 0.997 | 4,032 | 0.704 |
| bounded CaGR | 1.000 | 1.000 | 0.971 | 3,787 | 0.702 |
| **hard budget frontier** | **0.928** | **0.953** | **0.931** | **3,591** | 0.701 |

新方法在 elapsed、work 和 tail 上形成稳定优势，但 unique-compiled-pages 质量仍明显落后
frontier。这里不能写“所有三轴都 SOTA”：第三条 unique-page 轴揭示该方法优先完成便宜
query，而不是优先选择单位新页面最有证据价值的 query。frontier 也并非每个新域上的
strongest elapsed endpoint；它是预注册 reference，表中同时保留 bounded 的实际结果。

## 尾部与 ordering guarantee

五域所有 hard runs 的 max younger bypass 都精确达到 32，budget violation 为 0；硬约束
forced-selection fraction 约 27.6%--33.9%，不是未触发的装饰。P95/P99 的完整绝对值和
bypass P50/P95/P99 均在 JSON 中。

但 B32 仍只保证“至多被 32 个更年轻 query 越过”。它不自动约束绝对等待、slowdown 或
真实用户 tail-SLO。当前 P95/P99 数据是积极 transfer evidence，不是 B32 已解决所有用户
公平问题；后续仍需 joint age/slowdown constraint。

## 控制面开销

当前实现保留最透明的 exact cost 计算，每个 slot 重算 W64 候选并逐页模拟 virtual LRU。
五域 hard replay 的 deterministic detailed operation proxy 约为每 query 2,076--2,393 次，
跨 cell 平均约 2,265 次；它包含 utility evaluation、候选页 probe、hard-feasibility compare
和 event 数。

其他 baseline 目前只记录 dispatch+selection lower bound，因此 JSON 中约 1.13/query 的
数字不能和 hard method 的 detailed count 直接作倍数比较。本轮没有记录非确定性的 CPU
微基准，也没有 GPU throughput 数。结论是：控制逻辑在这些数百 query trace 上足够便宜
完成低成本 replay，但成为真实高吞吐系统前，需要增量 cost cache、feasibility frontier
或 heap 化，而不是声称开销已经解决。

## ColModernVBERT 缺失边界

本机可读的 artifact audit 记录了已生成过的 HR/Finance ColModernVBERT traces 及 hash，
但 manifest 指向 `/data/ldf/...`，当前主机没有原始 runtime/labels/cohort 文件。现有
`hr-k20.json` 等只含聚合 schedule，无法重放 arrival、B32 和 LRU，不能冒充 trace。

因此状态固定为 `not_run_missing_local_replay_trace`；没有下载、复制远端文件或调用 GPU。
这不会否定五域 ColPali/BM25-locator transfer，但论文 claim 必须写“跨域已验证，跨视觉
retriever 尚未验证”。

## 是否已经是论文主方法

按预注册 gate，它是 replay 级 **paper method candidate**，因为：

- causal interface 和 20/20 reference equivalence 同时成立；
- 五域全部可用，所有 cell parity/B32 safe；
- 三主轴无 ratio 超过 1.10，三项 median 都改善；
- 9/10 cell 有至少一个 5% 改善，10/10 P99 safe。

它比“解耦”或“前沿调度”更像具体 contribution：一个无标签、事件驱动、带 hard ordering
budget 的 completion/locality scheduler，并有从两域 oracle headroom 到五域 causal
transfer 的完整证据链。

仍未关闭的投稿风险是：跨 retriever、真实 GPU/compiler cost estimator、控制面优化和
joint tail constraint。最合理的下一步不是继续调 B/lambda，而是把完全冻结的方法接到
已有 ColModern trace 所在机器做 CPU-only replay，并选一个真实在线执行小实验校准 unit
cost。

## 审计与复现

- 预注册合同独立提交；
- policy interface/qrel invariance/no-end timer/W64 refill 均有小例测试；
- 五域本地输入逐 hash 校验，qrel/visual 只做 post-hoc quality；
- 主 JSON 保存 20-cell exact audit、五方法×五域×两 arrival×五 seeds 的完整三轴、P95、
  P99、bypass、cache、timer 和 operation counts；
- 全仓测试通过，连续两次分析输出 byte-identical；
- result JSON SHA-256：
  `afaa5b044bceb794ef8ef83630132356885f17467e184a3f4c39265fdc6fda15`。

复现命令：

```bash
PYTHONPATH=. python tools/analyze_causal_hard_frontier.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --domain-matrix-root /private/tmp/reprforge-vidore-domain-matrix-v1 \
  --domain-matrix-reference /Users/aura/gpu-systems-incubator/reprforge-worktrees/vidore-domain-matrix/results/diagnostics/vidore-v3-domain-matrix-v1.json \
  --modern-artifact-audit /Users/aura/gpu-systems-incubator/reprforge-worktrees/modern-retriever-transfer/results/modern-retriever-transfer/artifact-audit.json \
  --output results/systems/causal-hard-frontier-v1.json
```
