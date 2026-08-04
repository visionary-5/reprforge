# ReprForge 第三轮论文证据结论

日期：2026-08-04。基线分支：`research/anytime-index-base`。所有成功门均在读取对应留出结果前冻结。

## 结论先行

第三轮把当前工作从“两域上的有效启发式”推进成了一个可扩展、跨领域、跨视觉检索器成立的调度机制，但同时否定了两个更宽的论文故事：

1. **不能把贡献写成首次查询驱动、按需生成并跨请求保存索引。** EdgeRAG 已经覆盖这一宽泛生命周期。
2. **不能声称更早答对。** 三次独立答案实验均未通过预注册门，检索轨迹的改善没有稳定转化为答案正确性提升。

目前唯一值得继续投入的论文主线是：

> 给定固定的视觉多向量目标表示，廉价定位器把并发查询转成共享的查询—页面依赖图；ReprForge 调度下一批要构建的页面，使更多查询更早完成，并以原子版本发布可复现的增量索引状态。

这条主线仍是**有条件 GO**。投稿前必须正面击败 EdgeRAG 风格成本缓存和 CaGR-RAG 风格查询分组/预取；两条 P0 基线已经在独立分支启动。

## 主要实验结果

| 证据线 | 分支与提交 | 决策 | 最重要结果 | 论文作用 |
|---|---|---|---|---|
| 五域迁移 | `exp/vidore-domain-matrix` `22780c1` | **GO** | 新增 CS/Industrial/Pharma 相对 FIFO 分别少 18.2%/29.8%/21.6% 平均完成工作，且三域均优于热门度；12 个 K×B 点全部正收益 | 排除两域偶然性 |
| 现代检索器迁移 | `exp/modern-retriever-transfer` `972ceef` | **GO** | ColModernVBERT 在 HR/Finance K20 相对 FIFO 少 19.31%/28.74% 工作，AUC 均更高，最终语义完全一致 | 排除 ColPali 特有性质 |
| 控制面规模 | `exp/frontier-scale` `5cca66f` | **GO** | 216/216 组合完成；精确增量实现比 naive 快 17.45×；10K/338K 锚点低于 0.92 秒和 116 MiB；30K 压测低于 4.1 秒和 300 MiB | 证明调度器本身不会成为构建瓶颈 |
| 受限缓存/热启动 | `exp/capacity-warmstart` `8a5bcae` | **热启动贡献 NO-GO** | Finance/IR 冷转热只省 0.79%/3.30%，LRU 仍为 Belady 的 1.476×/1.471×；但相同有限缓存下 frontier 仍比 FIFO 少 30.4%/28.2% 工作 | 缓存是系统功能，不是独立贡献 |
| 证据到达代理 | `exp/answer-time-to-correct` `1fc5b43` | **仅代理 GO** | IRPapers gold 页首次持续到达 work：frontier 相对 FIFO 少 12.1%，只比热门度约好 0.7% | 只能称证据可用性，不是答案正确 |
| 自由答案与格式修复 | 同上 | **NO-GO** | 24 问 3B reader 的精确 gold 引用仅 4.2%；8 问 7B 格式修复只有 4/8 精确引用 | 排除“只修提示格式即可” |
| 结构化引用 | `exp/structured-citation` `20336f7` | **NO-GO** | 96/96 引用合法，但最终 Top-3 条件 gold 选择仅 46.7%，低于门槛；重选证据破坏原排序 | 系统映射解决接口，reader 二次选页失败 |
| 盲化语义答案 | `exp/answer-semantic-judge` `e2cb937` | **NO-GO** | initial/final 正确率同为 75%；正负修订各 2.1%；frontier 相对 FIFO 的语义 RM 改善 4.85%，未过 5%，且只由一条查询驱动 | 明确不能声称答案更早正确 |
| 最近工作审计 | `research/closest-work-audit` `56003ee` | **宽泛 novelty NO-GO** | EdgeRAG 覆盖粗定位、按需嵌入与跨请求缓存；CaGR-RAG 覆盖访问集合分组与预取 | 强制 retitle，并冻结两条 P0 判停基线 |

## 五域与两种表示的核心矩阵

旧 ColPali-v1.1 的冻结 K=20、B=8 结果：

| ViDoRe v3 域 | 查询×页面 | 事件复用率 | Frontier 相对 FIFO | Frontier 相对热门度 | 最终融合是否优于 BM25 |
|---|---:|---:|---:|---:|---|
| HR | 318×1,110 | 85.9% | -19.3% | -2.3% | 是 |
| Finance | 309×2,942 | 70.0% | -28.7% | -3.6% | 是 |
| Computer Science | 215×1,360 | 73.4% | -18.2% | -3.8% | 是 |
| Industrial | 283×5,244 | 62.4% | -29.8% | -5.0% | 是 |
| Pharmaceuticals | 364×2,313 | 75.3% | -21.6% | -2.8% | 是 |

ColModernVBERT 的跨表示结果：

| 域 | BM25 nDCG@10 | Modern K20 融合 | 旧 ColPali K20 融合 | Frontier vs FIFO | Frontier/FIFO AUC |
|---|---:|---:|---:|---:|---:|
| HR | 0.4860 | 0.5307 | 0.5373 | -19.31% | 0.5074 / 0.5011 |
| Finance | 0.5285 | 0.5790 | 0.5628 | -28.74% | 0.5657 / 0.5583 |

Modern 模型不是统一更准：它在 HR 略低于旧 ColPali，在 Finance 更高。稳定事实是依赖图和完成工作收益跨表示成立。

## 规模结果

精确增量 frontier 使用页面到查询的倒排表和惰性优先队列，保持原来的完整词典序 key 和批次语义。

- 18 个 Q=1K 精确对照、真实 HR 和 Finance 均与 naive 顺序摘要完全一致；
- 216/216 个 Q×pages×K×batch×分布组合完成，frontier 全部优于 FIFO 与热门度的平均完成工作；
- 对 naive 的聚合中位加速为 17.45×；
- Q=10K、pages=338K、K=20、B=32 的 uniform/Zipf/block 锚点为约 0.60--0.91 秒、92--115 MiB；
- Q=30K 可选压力点为 2.37--4.08 秒、205--299 MiB。

这些是调度控制面数据，不是假装成 338K 页面真实视觉编码。真实构建成本仍由 HR/Finance A100 和现代模型完整编码测量负责。

## 答案链路为什么失败

三层实验给出一致诊断：

1. **页面能更早到达。** gold-page proxy 相对 FIFO 有 12.1% 提前；
2. **接口可以被约束。** A/B/C/INSUFFICIENT 系统映射达到 100% 合法；
3. **模型并没有更可靠地使用新证据。** final 条件 gold 选择低于 initial，语义正确率没有上升，正负修订抵消。

后验诊断显示，永远引用检索 rank-1 的条件 gold 命中显著高于让 reader 二次选页。这说明当前最合理的产品设计是保留检索排序作为引用主干，只让 reader 判断支持/拒答；但由于该判断来自看过留出结果后的诊断，不能回写成本轮论文正结果。

## 新颖性边界

[EdgeRAG](https://arxiv.org/abs/2412.21023) 已经按查询生成缺失嵌入并缓存跨查询复用；[CaGR-RAG](https://arxiv.org/abs/2505.01164) 已经根据查询访问簇重叠分组并预取。因此以下表述全部删除：

- 首个查询驱动、惰性或在线 RAG 索引；
- 首次生成缺失嵌入并跨请求缓存；
- 首次利用查询重叠做调度；
- 首次用廉价文本定位器选择视觉工作。

可保留的是更窄、可被实验否证的表述：固定视觉多向量目标下，优化共享页面依赖的 cohort completion，而不是优化单请求 cache hit、已有向量的磁盘 I/O 或单查询 MaxSim。

## 下一步唯一 P0

已启动两个独立分支：

- `exp/edgerag-faithful-baseline`：高成本页面预建、成本×历史频率缓存、LRU/LFU、有限容量和漂移；
- `exp/cagr-faithful-baseline`：Jaccard 查询分组、组内连续执行、下一组预取，并计入预取浪费。

两者使用相同 HR/Finance、W64 burst/Poisson、五个排列、K=20、B=8 和容量约束。Frontier 必须在两个领域和设置中相对最强忠实基线至少改善 5%，且 P95 与质量遗憾不明显变差；否则停止把调度器作为论文主算法，转为 EdgeRAG 的视觉系统实现、measurement/benchmark 工作，或更换方法。

## 当前投稿判断

- **可以冻结的方法资产：** 依赖图形式化、resident-aware frontier、精确增量实现、原子发布与 anytime 指标。
- **已经足够强的数据：** 五域旧表示、两域现代表示、两域真实 A100、216 组合与 30K 控制面压力。
- **必须收缩的主张：** 不是端到端答案改进，不是一般低负载在线服务，不是在线索引生命周期首创。
- **决定是否继续当前论文的最后门：** EdgeRAG-faithful 与 CaGR-faithful。

建议标题：

> **ReprForge: Cohort-Completion Scheduling for On-Demand Visual Multi-Vector Index Construction**
