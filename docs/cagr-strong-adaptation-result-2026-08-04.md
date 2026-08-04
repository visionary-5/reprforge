# Strong CaGR-style adaptation 压力测试结果

日期：2026-08-04  
分支：`exp/cagr-strong-adaptation`  
预注册：`81c20bc`，实现冻结：`4dfb622`，审计元数据补丁：`2b62555`  
机器：CPU only，未使用 GPU  
机器可读结果：`results/systems/cagr-strong-adaptation-v1.json`

## 一句话结论

结论是 **STOP/DOWNGRADE**，但原因必须准确表述为：**81 个 CaGR-style adaptation 在 HR 的预注册部署资格门上全部失败，因此 strongest-adaptation 胜负不可证实**。这不是 Finance 上某个强 adaptation 反向击败 frontier，也不能写成 frontier 已经击败 strongest CaGR。

Finance 确实在三项配置冻结为 `null` 后才打开，并完成 FIFO、current overlap、faithful、frontier 等基线回放；由于没有 HR 合法部署项，Finance adaptation checks 为空。这是保守的不可证实结论，不是 vacuous GO。

## 预注册选择结果

选择阶段只打开 HR BM25 manifest 与 runtime，未打开 HR qrel、视觉分数或 Finance。HR 使用 burst/Poisson、五个固定 permutation seeds；81 个候选均保持最终 Top-20 union 为 895 页。

| 选择项 | 预注册候选范围 | 结果 |
|---|---:|---|
| lower-theta | 45 个 `theta<0.5` threshold 候选 | NO-DEPLOYABLE |
| fixed-size | 27 个 fixed Jaccard 候选 | NO-DEPLOYABLE |
| overall | 全部 81 个候选 | NO-DEPLOYABLE |

部署资格要求 burst 和 Poisson **同时**满足 singleton fraction ≤50%、query-slot utilization ≥50% 和 final-union parity。失败发生在 HR Poisson，不是参数排序阶段。

## 为什么 fixed-size 仍会退化

fixed-size 的含义是“从当前已经到达并等待的请求中，最多组成 4/8/16 大小的组”，不是等待未来请求直到凑满。回放又严格禁止跨组填充 batch。调度器只要发现 pending 非空就立即建组，因此在稀疏到达时，固定大小只是上限，不能保证下限。

观测与这一机制一致：

| HR 到达模式 | fixed-size singleton 范围 | batch 利用率范围 | 是否可能过门 |
|---|---:|---:|---|
| burst | 0.0% | 49.69%–99.38% | 部分候选可以 |
| Poisson | 60.51%–83.18% | 27.04%–33.79% | 全部不可以 |

作为对照，threshold family 在 HR Poisson 的最好 singleton 也只有 73.78%，最高利用率 26.45%。所以根因不是 `theta=0.5` 单点选错，也不只是 Jaccard 相似度过低，而是**在线可见请求不足 + 不等待 + 不跨组补批**的组合约束。

这个差异也解释了 burst 结果：burst 会制造足够 backlog，fixed-size 能形成接近满组；Poisson 下同一策略即时消费小 pending set，结构性退化。

## 最强但不具部署资格的 HR 诊断

以下配置仅按预注册 score 排名，用于解释失败，**不是已选择部署项，也没有进入 Finance gate**。

### Fixed Jaccard 最低 score

配置：target group size 16、pool 64、capacity 80；score=0.8889。

| 指标 | burst | Poisson |
|---|---:|---:|
| mean completion pages / FIFO | 0.9807 | 0.9848 |
| mean unit-cost / FIFO | 0.7721 | 0.8182 |
| singleton fraction | 0.0% | 83.18% |
| mean group size / max | 15.90 / 16 | 2.97 / 16 |
| batch utilization | 99.38% | 33.70% |
| cache hit | 33.87% | 26.10% |
| demand build / reload | 4,258 / 16,771 | 4,294 / 19,205 |
| prefetch build / reload | 217 / 1,075 | 181 / 733 |
| prefetch precision / wasted | 100% / 0 | 100% / 0 |
| starvation fraction / max bypass | 0% / 62 | 0% / 59 |

它说明 grouping 确实有成本潜力：相对 FIFO，mean unit-cost 下降 22.79%（burst）和 18.18%（Poisson）。但 Poisson 的 83.18% singleton 与 33.70% 利用率违反预注册部署条件，因此不能把这个数当作强适配胜负证据。

### Lower-theta 最低 score

配置：`theta=0.1`、pool 64、capacity 80；score=0.9153。

| 指标 | burst | Poisson |
|---|---:|---:|
| mean completion pages / FIFO | 0.9800 | 0.9823 |
| mean unit-cost / FIFO | 0.8324 | 0.8666 |
| singleton fraction | 59.97% | 79.37% |
| mean group size / max | 2.09 / 17 | 1.51 / 20 |
| batch utilization | 25.52% | 18.85% |
| cache hit | 65.86% | 52.36% |
| demand build / reload | 2,077 / 8,778 | 1,947 / 13,204 |
| prefetch build / reload | 2,398 / 11,066 | 2,528 / 7,943 |
| prefetch precision / wasted | 100% / 0 | 100% / 0 |
| starvation fraction / max bypass | 0% / 34 | 0% / 31 |

lower theta 减少了 faithful 的近全 singleton，但仍没有跨过部署门；大量精确 prefetch 虽全部最终有用，也产生了必须计费的 reload/build work，结果中没有删除这些工作。

## Finance 封存结果的正确用途

Finance 在配置摘要 `994c5f6d00c6b5cd98f35761612df8a1583743e2e6bf67d8549927df725bddfa`（三项均为 `null`）冻结后打开。它只能说明既有基线，不构成 strong adaptation 比较。

| Finance mean | faithful θ=.5 pages | frontier pages | frontier page 优势 | faithful θ=.5 unit-cost | frontier unit-cost | frontier cost 优势 |
|---|---:|---:|---:|---:|---:|---:|
| burst | 1,250.98 | 989.79 | 20.88% | 2,906.08 | 2,578.27 | 11.28% |
| Poisson | 1,250.76 | 1,004.26 | 19.71% | 2,903.11 | 2,658.14 | 8.44% |

这保留了 faithful `theta=.5` 对照下的正结果，但不能升级成“beat strongest CaGR”。另外 current overlap 的 Finance unit-cost 仍低于 frontier（burst 2,459.21 vs 2,578.27；Poisson 2,479.30 vs 2,658.14），说明 page-work 与 reload/build unit-cost 的 Pareto 张力仍然存在，论文不能只报页面完成指标。

## 判停解释

本次判停门没有被修改：

- gate decision：`STOP/DOWNGRADE`；
- `no_deployable_hr_selection=true`；
- Finance adaptation checks：0；
- 不能声称 Finance adaptation Pareto-dominates frontier；
- 也不能声称 frontier 对 strongest adaptation 保持 ≥5% 双指标优势。

因此当前可写的结论是：“faithful 映射在该页面访问图上近乎退化；预注册的 lower-threshold 与 fixed-size 强适配虽然改善抽象成本，但在开发域稀疏到达下无法同时满足成组质量与批利用率，strong comparison remains unresolved。”

## 对下一分支的直接启示

下一次不能在本门上回头放宽资格，而应另开预注册实验，把**什么时候允许等待更多请求**本身定义成调度决策。最小可检验设计是：

1. 队列深度触发：pending 达到 4/8 才成组；
2. 有限等待触发：最老请求最多等待固定 page-work budget，超时立即派发；
3. group-boundary fill：允许相邻组共同填满请求 batch，但保留组标签并单独统计跨组比例；
4. 目标同时加入 wait/sojourn，防止通过无限等待虚假降低 completion pages；
5. 仍只用 HR access graph 选择，重新封存 Finance。

这条新方向研究的是“到达稀疏度下，索引局部性收益与在线等待成本如何共同决定可成组性”，比继续扫 `theta` 更接近真正的调度/异构索引编译问题。

## 审计与复现

- HR 选择阶段打开文件只有 BM25 manifest 与 BM25 runtime；JSON 明确记录 `oracle_or_visual_files_opened=false`。
- HR/Finance 的 text runtime、visual runtime、oracle labels 与双 manifest 路径、字节数和 SHA-256 均写入 `inputs_after_unseal`。
- 每个聚合项记录五次 dispatch order 的 SHA-256、最终 union、build/reload/hit、prefetch、group、batch 和 starvation。
- 成本是 unit build/reload 模型，不是墙钟；未运行 GPU。
- 完整 81 候选表保留在 JSON 中，没有删除不利候选。
