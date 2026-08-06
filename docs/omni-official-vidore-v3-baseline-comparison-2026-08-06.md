# Omni 与官方 ViDoRe v3 流水线对照

日期：2026-08-06  
状态：HR 与 Computer Science 已完成同切分质量对照  
机器结果：`results/baselines/omni-vs-official-vidore-v3-pipelines-2026-08-06.json`

## 1. 为什么补这张表

当前实验已经证明 H-Pool、AGC、Full 之间存在明显的查询异质性，但这不等于 Omni
本身是最强检索器。为避免把系统机制收益和基础模型能力混在一起，本对照固定官方
ViDoRe 仓库提交 `a70f23af8bb3b33efe8a4a6c6c15a6e2d978035e`，在完全相同的
HR 与 Computer Science 公共切分上抽取官方流水线结果。

质量可以直接横向比较；官方时间由各提交者在不同硬件、批量和软件配置下报告，只能
说明数量级，不能与我们的 A100 原型时间直接相减。

## 2. Computer Science

| 方法 | nDCG@10 | Recall@100 | 官方搜索 ms/query |
|---|---:|---:|---:|
| mxbai edge ColBERT 32M | 0.6808 | 0.9439 | 4.1 |
| Qwen3 Embedding 8B | 0.7351 | 0.9736 | 45.9 |
| Nemotron Embed-VL 1B（图文） | 0.7391 | 0.9664 | 3.3 |
| **Omni H-Pool** | **0.7428** | **0.9826** | — |
| **Omni Full / Cascade-100** | **0.7475** | **0.9871** | — |
| **Omni AGC** | **0.7578** | **0.9802** | — |
| Nemotron Embed+Rerank 1B | 0.7805 | 0.8774* | 4,896.4 |
| Nemotron ColEmbed 3B | 0.7856 | 0.9873 | 680.1 |
| Nemotron ColEmbed 8B | 0.8007 | 0.9857 | 2,182.2 |
| Jina v4 + Zerank2 | 0.8346 | 0.9562 | 7,842.2 |
| Agentic 8B + Opus | 0.8449 | 0.8630 | 97,185.3 |

`*` 该 rerank 流水线只重排较浅候选，因而其 Recall@100 不能解释成底层检索器的完整
Top-100 覆盖能力。

Computer Science 上，Omni 高于廉价 32M、Qwen3 8B 和 1B 单向量流水线，但低于当前
强 late-interaction、reranker 和 agentic 流水线。这里存在可用的质量基础，却不存在
全局 SOTA 结论。

## 3. HR

| 方法 | nDCG@10 | Recall@100 | 官方搜索 ms/query |
|---|---:|---:|---:|
| Qwen3 Embedding 8B | 0.5233 | 0.8908 | 40.3 |
| mxbai edge ColBERT 32M | 0.5261 | 0.8722 | 4.0 |
| **Omni H-Pool** | **0.5503** | **0.9106** | — |
| **Omni Full** | **0.5704** | **0.9343** | — |
| **Omni AGC** | **0.5782** | **0.9112** | — |
| Nemotron Embed-VL 1B（图文） | 0.6080 | 0.9276 | 3.6 |
| Nemotron ColEmbed 3B | 0.6548 | 0.9432 | 455.3 |
| Nemotron Embed+Rerank 1B | 0.6573 | 0.7919* | 6,169.2 |
| Nemotron ColEmbed 8B | 0.6841 | 0.9462 | 1,146.3 |
| Jina v4 + Zerank2 | 0.6910 | 0.9416 | 9,785.3 |
| Agentic 8B + Opus | 0.7450 | 0.7535 | 101,714.7 |

HR 暴露了基础模型代际差距：Omni 只高于两条廉价文本/嵌入基线，低于当前 1B 视觉
嵌入和所有强多向量、重排流水线。因此“优化 ColPali v1.1 检索质量”不应成为论文主张。

## 4. 对论文问题的含义

这张表没有否定 representation compiler，反而把它从模型贡献中剥离得更清楚：

1. 当前公共流水线横跨约 3 ms/query 到 97 s/query，质量也不单调随成本变化，真实系统
   确实需要管理多种能力和成本路径；
2. 我们已有的两域证据说明，压缩表示造成的损失多数属于候选内排序不足，可由按需 Full
   打分修复；
3. 但若只在旧 checkpoint 上成立，审稿人可以把贡献判断为模型特例。因此最终机制必须
   迁移到至少一个当前强模型族。

论文的正确主张应是：**给定一个基础检索器及其多种物理表示，查询感知编译器在质量约束
下减少构建、存储和访问成本，并明确何时需要升级到高保真表示或第二定位器。** 它不是
重新训练一个全局最强检索器。

## 5. 下一步实验优先级

1. 完成 Finance，随后在磁盘允许时运行 Pharmaceuticals；用四域左右的数据确认
   “定位失败与排序失败”的比例是否稳定。
2. 将同样的 Full / cheap locator / candidate rerank 路径迁移到 Nemotron ColEmbed 3B，
   或先用官方冻结排名做模型无关回放。成功标准不是超过其 Full nDCG，而是在预设质量
   容差内降低冷层访问量和生命周期成本。
3. 接入第二套 benchmark（优先 REAL-MM-RAG），排除只适配 ViDoRe 标注方式的可能。
4. 在上述证据形成前，不训练新的 query policy；风险信号和阈值只做冻结的离线诊断。

## 6. 可复现性边界

机器结果记录了每个官方 JSON 与 Omni 审计 JSON 的 SHA-256、官方仓库 revision、两域
逐方法均值、索引字节数和官方自报时间。官方时间保留为外部上下文，不用于计算本系统
相对加速比。
