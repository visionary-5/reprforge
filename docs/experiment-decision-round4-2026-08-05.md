# ReprForge 第四轮实验决策

日期：2026-08-05

## 一句话结论

这一轮没有找到新的通用评分器、工作负载选择器或长周期重优化器，但把论文主线收得更实：

> ReprForge 的贡献应是异构表示的 exact build+reload 物理状态、严格因果的在线编译合同、保持决策完全一致的增量控制面，以及 elapsed、charged work、unique-page quality 三轴测量；Causal Hard Frontier 是冻结的系统策略，不是新的公平或评分算法。

## 分支决策

| 分支 | 结果提交 | 注册结论 | 最重要的数据 | 集成决定 |
|---|---|---|---|---|
| `exp/causal-hard-frontier` | `29448fd` | **PAPER METHOD CANDIDATE，算法新意受限** | 20/20 reference exact；五域 median sojourn/work/elapsed 0.923/0.949/0.948；10/10 P99 safe | 集成核心 policy、replay、测试和主结果，作为因果编译载体 |
| `exp/b32-scoring-ablation` | `f5b04c1` | **SCORING CONTRIBUTION NOT SUPPORTED** | 三主指标同向 3/10；median geometric mean 1.0009；明确 tail win 0/10 | 只吸收论文结论；不合并 NO-GO 方法代码和 5.83 MB JSON |
| `exp/incremental-causal-control-plane` | `b375224` | **INCREMENTAL CONTROL PLANE GO** | 五域 50/50 exact；operations 12.63×；CPU 2.084×；peak memory ratio .971；与 efficient Delay 接近 | 集成代码、测试、报告和小型结果；作为系统实现贡献 |
| `exp/workload-physical-plan-selector` | `51014c3` | **ORACLE SPACE GO / DEPLOYABLE PROXY NO-GO** | oracle 综合比 .961；LODO selector 1.001，24 次 SLO violation | 只吸收负结果；不合并 selector 代码和完整 JSON |
| `exp/adaptive-compile-reoptimization` | `44c81d1` | **LONG-HORIZON REOPTIMIZATION NO-GO** | 带成本 oracle 仅改善 .227%；免费 oracle 仅 .637% | 只吸收负结果；不合并 reoptimizer、trigger 或完整 JSON |
| `exp/multilevel-representation-compiler` | `88bf45d` | **CURRENT ARTIFACT NO-GO / QUERY ORACLE HEADROOM** | Finance/Industrial query oracle nDCG@10 分别 +.1105/+.1227；三个真实 route 都被选择 | 集成小型 capability audit、合同和结果；不能声称动态 compiler 已完成 |
| `exp/prefix-feedback-plan-selection` | `39d1fbe` | **HEADROOM NO-GO** | 50/50 continuation exact；1/50 tail gain=-.0040127，注册 normalized regret 无定义；oracle/LODO 未运行 | 只吸收合同结论；代码和 JSON 留分支，不能事后更换质量分母 |

## 新数据怎样改变论文

### 1. 评分路线已经关闭

completion-only 捕获了大部分系统收益；加入 continuous age 后没有跨域稳定的 2% 效果或 tail 改善。age-only 明显失败。可以保留的组件发现是：completion normalization 有用，而忽略 active-LRU reload 的 build-only proxy 会让系统指标中位恶化约 19%--23%。这支持 exact physical-state accounting，不支持新的 `.75/.25` 评分公式。

### 2. 控制面路线已经过门

增量实现没有改近似、候选集合或 tie-break，而是删除每候选 compiled-set copy、重复 cohort sort 和 all-older feasibility scan。它在 50 个完整 replay 中与 reference 的 dispatch、completion、work、cache、bypass 和 publication trace 全部一致。Hard 与 efficient Delay 的控制面已基本同成本，因此论文可以公平比较系统策略，不需要为透明 Python 实现的低效辩护。

### 3. 自动选计划和长期重优化都没有可部署空间

工作负载 selector 的 oracle 主要通过提前发布 post-hoc 高价值 evidence 获益，但上一 episode 的 locator graph 不能跨域预测这种价值。长期 stress 中，B32 和 Delay 只是在很小 margin 内交换第一名；即使免费 oracle 也不到 1%。继续加 learner、trigger 或权重只会扩大搜索空间，不会形成当前论文贡献。

### 4. 多级表示仍值得做，但必须先补 artifact

Finance 和 Industrial 的真实 text、pool-4、full surface 都显示明显 query heterogeneity，而且三个 route 都会被 oracle 选中。这说明表示不是单调“越大越好”的档位，而更像共享编码前缀和派生 view 组成的 build DAG。

现有 bundle 仍缺四样关键东西：

1. query-scoped mixed-document activation outcome；
2. query 到 item 的真实 activation stream；
3. 同 workload 的逐 item build 和 compact-derivation 成本；
4. cold/warm reload、H2D、bytes 与 parent lineage。

缺这些数据时运行 LRU、LFU、GDSF 或 dynamic oracle 会制造跨数据集 proxy，不能算结果。

### 5. Prefix feedback 不能用当前归一化指标启动 selector

prefix 实验先验证了 50/50 continuation replay exact，但有一个 cell 的 continuation tail gain 为
`-0.0040127`。注册指标把后续 regret 归一化到正的 tail gain；分母为负时该量不再表示“剩余可获得
质量的比例”，因此按合同在 headroom 阶段直接 NO-GO，oracle 和 LODO selector 都没有运行。

这不是“prefix observables 已被证明没有预测力”，而是更基础的测量发现：渐进检索质量可以
非单调，截断到 continuation 后尤其不能假设 tail gain 为正。若未来重启，必须另开预注册，采用
允许正负变化的 signed 指标，或回到完整 episode 的统一分母；不能看见这个 cell 后临时修改 gate。

## 下一步最小实验

下一轮不做新 scheduler，也不训练 selector。只做一个可以快速关闭或打开多级 compiler 方向的物理测量：

1. 扩展 MMDocIR 的 query-scoped pool/full 执行，保存每步 locator candidate 与 activation IDs；
2. 保存 pool-only 和任意 retained-full subset 的可比较检索结果，避免全局分数污染；
3. 在同一 A100、同一 workload 测逐 item full encode、compact derivation、cold/warm reload、H2D、bytes 和 batch overlap；
4. 保存 representation lineage，包括共享 encoder prefix、派生 view、保留 parent 和重建 parent 的成本；
5. 至少使用两个冻结 MMDocIR 文档角色或 workload stream；
6. 先跑 uniform、full-eager、transient、GDSF、static oracle 和 dynamic oracle。

只有在两个角色上，oracle 都能在 matched nDCG@10 下相对最强 full-eager、transient 和 GDSF 节省至少 10% charged build+reload+storage cost，才继续设计在线 compiler。若 oracle 不过门，应直接关闭多级动态控制器，把现有工作固化为在线索引编译系统与测量论文。

## 当前投稿边界

可以写：

- exact build+reload/LRU physical state 会实质改变调度成本；
- causal policy 不读取 qrel 或未来 trace；
- 增量实现能在保持 50/50 full-trace exact 时显著降低控制面成本；
- 三条质量时间轴揭示真实而非单一 winner 的系统 Pareto；
- 多级真实 surface 存在 query-level headroom，但部署 artifact 尚不完整。

不能写：

- B32、bounded skip 或 completion+age 是新的调度算法；
- selector 或长期 reoptimizer 已经可部署；
- 已完成多级动态表示 compiler；
- unit/additive page cost 等于真实并发 GPU wall-clock；
- 检索证据提前已经转化为答案更早正确。
