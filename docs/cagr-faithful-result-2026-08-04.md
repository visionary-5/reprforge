# CaGR-RAG 忠实基线结果

日期：2026-08-04<br>
预注册：`docs/cagr-faithful-preregistration-2026-08-04.md`，commit `50f78e6`<br>
论文来源：[CaGR-RAG: Context-aware Query Grouping for Disk-based Vector Search in RAG Systems](https://arxiv.org/html/2505.01164v1)

## 决策

**CPU completion-page novelty gate：GO。完整 measured-cost / 墙钟结论：尚不可用。**

在冻结的 HR/Finance、K20、W64、burst/Poisson、batch 8、五个 permutation 上，frontier 相对 CaGR-faithful 的 mean completion page-work 分别改善 14.5%、7.2%、20.9%、19.7%。四个设置的 normalized quality regret 都更低，P95 completion page-work 均不超过 CaGR 的 1.05 倍，因此通过预注册的四项合取门。

这个结果支持一条窄结论：**当前 frontier 的页面完成目标不是 CaGR-RAG Algorithm 1 的阈值查询分组与组切换预取换名。** 它不支持“frontier 是一般最优在线调度器”：离线 full-stream static popularity 距 frontier 只有 1.5%–4.2%，frontier 仍有 11.5%–23.8% 的严格饥饿，而且缺少可分辨 reload/I/O 与 batch overhead 的真实成本 profile。

## 主门结果

| 领域 / 到达 | CaGR mean / P95 pages | Frontier mean / P95 pages | Frontier mean 改善 | CaGR / Frontier 质量遗憾↓ | CaGR / Frontier unit-cost mean↓ | Gate |
|---|---:|---:|---:|---:|---:|---|
| HR / burst | 732.8 / 891 | **626.9 / 854** | **14.5%** | 0.8071 / **0.6879** | 2868.9 / **2491.7** | GO |
| HR / Poisson | 733.2 / 891 | **680.3 / 891** | **7.2%** | 0.8096 / **0.7396** | 2921.5 / **2620.1** | GO |
| Finance / burst | 1251.0 / 1819.4 | **989.8 / 1698** | **20.9%** | 0.6769 / **0.5022** | 2906.1 / **2578.3** | GO |
| Finance / Poisson | 1250.8 / 1817 | **1004.3 / 1713** | **19.7%** | 0.6774 / **0.5129** | 2903.1 / **2658.1** | GO |

`unit-cost` 严格按预注册记账：build=1、compiled 但 active-cache miss 的 reload=1、hit=0，预取工作不免费。它不是毫秒或墙钟。按这个抽象成本，frontier 相对 CaGR 的 mean completion cost 也改善 8.4%–13.1%，P95 cost 更低 8.8%–14.1%。由于 build 和 reload 被设成相同单位，这只能说明结论不依赖“只数唯一构建页”这一种记账，不能替代物理测量。

四个 setting 的最终候选并集分别保持 HR 895 页、Finance 1855 页；最终完成查询数和冻结最终质量相同。frontier 的 W64 completion pages 与先前 `windowed-arrivals-v1.json` 四个对应单元逐项一致。

## 全基线 completion page-work

| 领域 / 到达 | FIFO | 当前 causal overlap | 历史热门度 | 离线 static popularity | CaGR-faithful | Frontier |
|---|---:|---:|---:|---:|---:|---:|
| HR / burst | 743.1 | 726.9 | 717.3 | 654.2 | 732.8 | **626.9** |
| HR / Poisson | 742.4 | 732.6 | 710.7 | 690.7 | 733.2 | **680.3** |
| Finance / burst | 1272.9 | 1221.1 | 1146.7 | 1016.5 | 1251.0 | **989.8** |
| Finance / Poisson | 1272.1 | 1227.5 | 1136.5 | 1027.6 | 1250.8 | **1004.3** |

static popularity 使用完整流的页面频率，只在最老 W64 pending 内选择，因此是明确标注的 offline diagnostic。frontier 对它只改善 4.2%、1.5%、2.6%、2.3%；这比对 CaGR 的领先小得多。安全说法是“frontier 通过 CaGR novelty gate”，不能写“已排除 popularity oracle 或所有简单调度解释”。

当前 overlap 在 completion pages 上明显弱于 frontier，但在 unit-cost 上反而更低：HR burst/Poisson 为 2428.9/2548.3，对应 frontier 2491.7/2620.1；Finance 为 2459.2/2479.3，对应 frontier 2578.3/2658.1。这来自 overlap 对容量 80 active cache 的局部性更好，说明**优化唯一构建页完成顺序与优化 finite-cache reload 并不是同一目标**。在物理 I/O/重载占主导的系统里，frontier 可能不是赢家；这是下一次真实成本实验必须回答的问题。

## CaGR 忠实性诊断

主实现遵循论文 Algorithm 1：访问集 Jaccard、`theta=0.5`、到达顺序扫描、加入第一个与任一 group member 达阈值的 group、group 创建顺序执行、组内保持到达顺序，每个 group 结束后预取下一 group 第一条查询的精确 Top-20 访问集。

| 领域 / 到达 | CaGR cache hit | Prefetch precision | Singleton groups | 最大 group | Query batch slot utilization | 饥饿 |
|---|---:|---:|---:|---:|---:|---:|
| HR / burst | 96.6% | 100% | 98.8% | 4 | 12.7% | 0% |
| HR / Poisson | 71.8% | 100% | 99.2% | 3 | 12.6% | 0% |
| Finance / burst | 97.0% | 100% | 98.6% | 2 | 12.7% | 0% |
| Finance / Poisson | 96.3% | 100% | 98.4% | 2 | 12.7% | 0% |

prefetch precision 为 100% 不是预测能力：算法预取已知“下一组第一条查询”的精确集合，并在下一组立即需求，所以容量至少为 K20 时自然全部使用。所有预取 build/reload 都已进入 unit-cost；unused prefetch work 为 0。真正有信息的结果是 group fragmentation：页面候选集的 Jaccard 远比 IVF cluster access set 稀疏，几乎所有 group 都是 singleton。因此 CaGR 大多退化成保序逐组执行加精确预取。

诊断中的 pool 20/40、cache 40/160，以及按 Equation 3 使用 `all members` 的严格规则，均没有改变主判断：mean completion pages 的差异很小，最大 group 仍为 2–4。它们未进入 GO gate。

## 质量、尾部和公平性

frontier 的质量遗憾在四个设置均低于 CaGR，quality--work AUC 也更高。原因不是调度器读取了质量：调度 API 不接收 qrel、视觉分数或 quality gain；这些值只在批次选择完成后累计。测试还用交换后的 quality gain 证明 dispatch order 不变。

但平均完成收益以公平性为代价：frontier 严格 starvation fraction 为 HR burst 23.4%、HR Poisson 11.5%、Finance burst 23.8%、Finance Poisson 22.7%；CaGR 保持到达/group 创建顺序，四项均为 0%。当前 gate 只要求 regret 和 completion P95，没有把 starvation 设为否决项，所以形式上仍为 GO；论文必须继续保留“深队列批量负载、有尾部公平性问题”的边界。

## 为什么没有启动真实 GPU

仓库已有 A100 `b8-resident` artifact 可给出混合 build/scoring 路径的粗诊断：HR 97.80 s / 895 页约 109.28 ms/page，Finance 189.89 s / 1855 页约 102.37 ms/page。但这些汇总没有独立的 compiled-page reload、磁盘/主存读取、每批固定 overhead 或异步预取重叠测量，无法忠实重放 CaGR 的物理目标。

因此本分支把 `measured_cost_profile.available` 明确记为 false，没有拿聚合总时间反拟合 reload 参数，也没有占用共享 GPU。直接按当前顺序重跑视觉 encoder 只能验证“不同 query batch 造成的构建时间”，不能验证 CaGR 的 disk-vector cache/prefetch 机制。合理的 GPU/系统实验应先实现可区分的 build、persistent-index load、active-cache hit、异步 prefetch 计时，再冻结运行；否则会用真实硬件测一个不忠实的问题。

## Fidelity gaps

1. Top-20 页面候选集代替 IVF nprobe=10 cluster set，是本实验最大的迁移差异，也是 singleton 比例极高的原因。
2. CaGR 原对象是已构建向量的磁盘加载；本实验显式拆成不淘汰 `compiled` 与容量 80 `active_cache`，但 CPU unit-cost 仍不是 NVMe/FAISS 延迟。
3. 原论文随机使用 20–100 query batch；主实验固定现有 W64 到达窗口，20/40 只做诊断。
4. 原论文逐查询搜索；ReprForge 最多 8 个 query 原子发布。为保持 group-contiguous，CaGR 不跨 group 补齐 request batch，导致约 12.6% slot utilization。这个数必须和物理结果一起披露，不能把低利用率藏起来。
5. 截至 2026-08-04，未定位到作者官方 CaGR-RAG 代码；实现依据论文 Algorithm 1。论文 Equation 3 的 `all` 与伪代码 `max` 歧义已用主/诊断两套规则覆盖。

## 结论边界与下一步

本 P0 的结果是 **conditional GO**：

- 可以写：frontier 在共享视觉页面的 completion objective 上，稳定优于 CaGR Algorithm 1 的直接适配；两者优化的对象不同。
- 不能写：frontier 已在真实延迟上击败 CaGR，或已击败所有 locality/popularity scheduler。
- 下一步优先级：实现带真实 load/reload/prefetch 的两层物理 replay，正面对比 current overlap；如果 overlap 在物理成本上保持优势，应把方法改成 completion/locality 的多目标 scheduler，而不是继续强化单一 frontier 分数。

## 复现

```bash
python -m tools.analyze_cagr_faithful \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --output results/systems/cagr-faithful-v1.json

pytest -q tests/test_cagr_faithful_replay.py \
  tests/test_windowed_arrival_replay.py \
  tests/test_scheduler_baselines.py
```

机器可读结果：`results/systems/cagr-faithful-v1.json`。定向测试：16 passed。
