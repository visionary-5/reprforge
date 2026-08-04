# Hard younger-bypass constrained oracle 结果

日期：2026-08-05

分支：`exp/hard-fair-oracle`

预注册：`060d131`

机器：CPU only，未使用 GPU

机器可读结果：`results/systems/hard-fair-oracle-v1.json`

## 一句话结论

预注册判定为 **HARD-FAIR HEADROOM GO**：固定上一轮最强松弛效用后，只扫描
younger-bypass budget `B∈{8,16,32,64}`，HR 唯一合格点为 `B=32`；该配置冻结后在
Finance burst 与 Poisson 都保持 union parity、零预算违规、零 `bypass>=64`
starvation，并同时改善 mean sojourn 与 charged work/query。

这里的 “hard-fair” 只表示一个可证明的调度顺序性质：任何 query 最多被 32 个更年轻
query 越过。它不等于已经解决用户感知公平性。并行审计显示 bypass 与 extreme
sojourn/slowdown 的关联可能只是 weak/partial，后续仍需联合 age、slowdown 和 tail 的
transfer audit。

## 方法：固定效用，只改变硬预算

沿用上一轮 `oracle_15`，不再调软权重：

- quality weight 0，completion weight 0.75，deadline weight 0.25；
- deadline scale 256，future-wait timeout 16；
- 每个 batch 槽基于已到达 query 的 locator cohort 和当前 virtual compiled/LRU，精确
  模拟下一 query 的 demand cost；
- 不 prefetch、不提前服务尚未到达 query，batch max 8、LRU cap 80、unit clock 不变。

每选择一个较年轻 query，就立刻给仍 pending 的更老 query 增加一次 bypass。会使任一
更老 query 超过预算 `B` 的候选不可选；最老 pending query 永远可行，因此没有死锁。

## HR 选择结果

下表 ratio 越低越好。mean sojourn/work 对 bounded CaGR，elapsed regret 对 frontier，
P95 对 bounded/frontier 中较好的 endpoint。

| B | max primary ratio burst / Poisson↓ | P95 ratio burst / Poisson↓ | starvation burst / Poisson | forced slot burst / Poisson | 资格 |
|---:|---:|---:|---:|---:|---|
| 8 | 1.162 / 1.178 | 1.167 / 1.182 | 0% / 0% | 69.25% / 67.80% | fail |
| 16 | 1.059 / 1.077 | 1.078 / 1.088 | 0% / 0% | 53.40% / 52.58% | fail |
| **32** | **0.950 / 0.956** | **0.953 / 0.951** | **0% / 0%** | **30.38% / 30.75%** | **pass** |
| 64 | 0.871 / 0.868 | 0.873 / 0.835 | 12.64% / 12.39% | 12.33% / 12.01% | fail |

`B=8/16` 太紧，强制回退过多，时延、工作量和 P95 反而超过 endpoint；`B=64` 保留更多
局部性收益，但冻结定义把 `bypass>=64` 计为 starvation，因此不合格。`B=32` 是唯一
qualified 点，也是唯一 Pareto 点，配置 SHA-256 为
`79b6aa7a5ed353bd19cb07109e45893b441835f1f16020c52b208c876b0b41f6`。

这比“解耦”更具体：数据表明软 deadline 权重无法形成安全插值，而硬 ordering budget
产生了一个非平凡可行区间；过紧、可行、过松三个区域都被四点实验观察到。

## Finance 冻结迁移结果

这里的 “冻结” 只表示配置选择函数没有 Finance 数据依赖，代码路径先在 HR 选择并固定
SHA，随后才读取包含 Finance evaluation 的 time reference 和 Finance workload，整个
transfer 不做 Finance 调参。仓库历史中早已有 Finance endpoint 与相关实验，因此不能
声称实验者层面的全新盲测或严格未开封测试。

### 绝对指标

| arrival | 方法 | mean sojourn↓ | P95 sojourn↓ | work/query↓ | elapsed regret↓ |
|---|---|---:|---:|---:|---:|
| burst | **B32 oracle** | **2,026.69** | **3,872.0** | **14.913** | **0.40554** |
| burst | bounded | 2,204.44 | 4,114.0 | 15.829 | 0.41895 |
| burst | frontier | 2,302.32 | 4,541.4 | 16.199 | 0.40779 |
| Poisson | **B32 oracle** | **1,740.46** | **3,353.20** | **14.957** | **0.40737** |
| Poisson | bounded | 1,886.81 | 3,574.32 | 15.762 | 0.41234 |
| Poisson | frontier | 2,041.67 | 4,365.72 | 16.477 | 0.42311 |

### 相对改善与硬约束触发

| arrival | mean sojourn vs bounded | work/query vs bounded | elapsed regret vs frontier | max bypass | forced slots | protected query fraction | 预算违规 |
|---|---:|---:|---:|---:|---:|---:|---:|
| burst | **8.06%** | **5.79%** | 0.55% | 32 | 494/1545 = 31.97% | 32.30% | 0 |
| Poisson | **7.76%** | **5.11%** | 3.72% | 32 | 466/1545 = 30.16% | 30.42% | 0 |

两种 arrival 都通过全部预注册约束，并各有至少一个主轴改善不低于 5%。Poisson 有 5
次 future-wait、总计 80 unit-time、单次最多 16；burst 没有触发。强制改选约占三成
slot，说明收益不是一个从未生效的约束造成的偶然结果。

## 三条质量时间轴不能混写

Finance 上 B32 的 elapsed regret 为 0.40554/0.40737，优于 frontier；charged-work
regret 为 0.40554/0.40462。但 unique-compiled-pages regret 为 0.66055/0.65917，明显差于
frontier 的 0.50257/0.51316，甚至略差于 bounded 的 0.65377/0.63871。

因此不能写成“所有 anytime quality 曲线都更好”。准确结论是：在真实 elapsed 与
charged work 轴存在收益，按 unique page 计量的早期质量仍是弱项，曲线会交叉。这个
差异也提示下一版 design 不能只优化局部性和完成成本，还需加入不使用 qrel 的证据价值
proxy。

## 实际可观测性审计

预注册为避免事后放宽，仍把整个有限 oracle family 称为 clairvoyant probe；但事后检查
发现被选中的 B32 策略比 family 名称更接近在线策略：

- `lambda_quality=0`，dispatch 不读取冻结 qrel gain；交换 qrel gain 的单元测试得到完全
  相同的 dispatch 与 charged work；
- exact next-demand cost 只依赖已经到达 query 的 locator cohort、当前 compiled set、
  当前 active LRU，以及 batch 内已经选择的 query；不读取未来 query 内容；
- deadline 和 B32 只依赖当前时钟、arrival rank 与在线 bypass counter；
- future wait 可由“为最老 pending query 设置 16-unit timer，arrival event 触发时重算”
  等价实现，不必知道下一 arrival 的时间或内容。

唯一差别在有限 replay 尾部：注册实现知道数据集已经结束，会立即 flush 未满 batch；没有
end-of-stream/session 信号的在线服务应继续等到已设置的 deadline。额外 post-hoc 诊断把
末尾改成后一种保守语义，且不参与 HR 选择、注册 horizon 或主 GO。Finance 10 个
arrival×seed replay 中，dispatch order、charged work、final union 和完成 elapsed 都逐项
完全一致，重新计算同一组门槛仍为 GO。小例测试同时覆盖了真正存在尾部差异时只增加
等待、不改变顺序与 work 的情况。

所以更精确的表述是：**名义实验是有限 constrained-oracle headroom；被选中的 dispatch
rule 在本模拟状态接口下是 causal 的，future wait 也可事件驱动在线实现，但它还不是经
真实系统验证的 deployable scheduler。** exact cost simulator、locator 可用时刻和真实
GPU 并发成本都仍需落地校准。

## 能说什么，不能说什么

本轮支持：

1. relaxed `oracle_15` 的主要 headroom 可以在严格 `B=32` ordering guarantee 下保留；
2. 该配置从 HR 唯一选择后迁移到 Finance，两种 arrival 都得到 mean/P95 和 work 改善；
3. hard constraint 与 completion/locality utility 的组合是值得蒸馏的 design 方向。

本轮不支持：全局最优、所有异构索引 workload 都迁移、用户感知公平性已经解决，或真实
GPU wall-clock 已改善。下一步应把 B32 与绝对 age/slowdown 或 tail-SLO 做 joint
constraint，并在更多 benchmark、真实编译时间和并发执行下做 transfer audit。

## 审计与复现

- 四个 endpoint 的全部 system aggregates/order hashes 与 time reference 精确一致；
- 所有 hard runs final union/query parity，budget violation 为 0；
- 配置选择函数不接收 Finance 数据；经测试的执行顺序为 HR load/evaluate/freeze、读取
  Finance-bearing reference、加载 Finance workload；
- 历史 Finance endpoint 已存在，本实验只称 frozen transfer/no Finance tuning，不称
  human-level blind test；
- 三个主门分别使用预注册的 strongest endpoint：sojourn/work 对 bounded CaGR，elapsed
  quality regret 对 frontier；unique-page quality 则明确落后；
- JSON 保存每 seed 三轴、system、trigger、dispatch/trace hashes、输入 provenance、主 gate
  和独立 observability diagnostic；
- 全仓测试通过，JSON 连续运行 byte-identical。
- result JSON SHA-256：
  `765fbcdad3d02a2cb1a1222f87874f3721a797c933f226051df8a04d61e62233`。相对首版
  `a53d190e...fc2f` 只修正 selection/transfer provenance 字段；删除这些字段后其余 JSON
  精确一致，所有数值、顺序、GO 与 observability 结果不变。

复现命令：

```bash
PYTHONPATH=. python tools/analyze_hard_fair_oracle.py \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --time-reference results/systems/time-aligned-quality-v1.json \
  --output results/systems/hard-fair-oracle-v1.json
```
