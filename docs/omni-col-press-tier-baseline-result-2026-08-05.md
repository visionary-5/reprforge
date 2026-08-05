# OmniColPress Full/H-Pool A100 smoke 结果

> 日期：2026-08-05  
> 判定：**GO_TO_SCALE**；仅验证真实表示层级和实验链路，不报告为论文质量结果。

## 1. 结论

同一冻结 `ColBERT_qwen2.5-vl_colpali` 模型下，OmniColPress 的参数无关 H-Pool
确实形成了可物化的第二级表示：每页从 1,406 个 2,048 维向量压到固定 64 个，完整
索引目录从 368,943,239 bytes 降到 16,784,998 bytes，缩小 **21.98×（-95.45%）**。
在这份 16-query smoke 上，Recall@1 完全保持；Recall@5 损失 6.25 个百分点，MRR
损失 1.56 个百分点。这证明 Full→H-Pool 是 ReprForge 可以真实构建、持久化、加载和
计价的表示边，但不证明 64 向量配置在完整 benchmark 上安全。

## 2. 冻结输入与运行环境

- 输入：ViDoRe v3 HR 的 32 页、16 条英文查询和完整 qrel 子集；选择策略偏向新增
  正相关页较少的查询，因此不能外推总体质量。
- 官方源码：`omni-col-press@4a559677bbc8a3ea0c10322a721b52bb70d382ec`。
- 冻结模型：`hltcoe/ColBERT_qwen2.5-vl_colpali@14a7bb3328187705ff153e3511a47f9abb144054`。
- ReprForge wrapper：`470bcd4`；单张 NVIDIA A100-SXM4-80GB，物理 GPU 3。
- 软件：Torch 2.5.1、Transformers 4.57.1、Torchvision 0.20.1+cu124，SDPA。
- 服务器原始输出：`/data/ldf/reprforge/experiments/strong-baselines-20260805/outputs/omni-hr-full-hpool-470bcd4-gpu3`。

运行器记录四个公开兼容补丁的 SHA-256。两处只延迟导入本次不用的 Fast-Plaid；两处
只把布尔 attention mask 搬到已有计算张量的设备，不改变掩码值或检索方法。

## 3. 真实结果

| 方法 | 索引张量形状 | 完整索引目录 bytes | 构建 wall time | Recall@1 | Recall@5 | nDCG@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 32×1,406×2,048 fp32 | 368,943,239 | 55.86 s | .8750 | 1.0000 | .94135 | .921875 |
| H-Pool-64 | 32×64×2,048 fp32 | 16,784,998 | 65.49 s | .8750 | .9375 | .91443 | .906250 |

文件大小来自服务器 `stat`，张量 payload 来自实际 `shape × element_size`，不使用
OmniColPress 的估计函数。H-Pool 构建比 Full 多 9.63 秒（+17.24%），符合额外层次聚类
开销。端到端评价分别为 16.66 和 19.86 秒，但 16-query 运行主要由模型加载主导，
不能据此声称压缩索引检索更慢；规模实验需要拆开模型加载、query encoding 和 search。

## 4. 对论文问题的含义

这不是“我们发明了 H-Pool”。它给 ReprForge 提供了真实的物理层级：Full 是高质量、
大体积状态，H-Pool-64 是便宜常驻状态。论文问题应继续验证：编译器能否按文档或
工作负载风险选择层级，使整体索引接近 H-Pool 的存储/构建成本，同时把竞争边界上的
高风险页面升级到 Full，从而优于任何固定压缩率。

下一轮必须使用完整公开域并报告 paired per-query regret、bootstrap 置信区间、
Top-k 边界翻转、实际索引字节和拆分后的构建/查询时间。固定 Full、固定 H-Pool、随机
等预算升级和 qrel-only oracle 是最低比较组；smoke 指标不进入主表。

## 5. 制品

机器可读摘要见
`results/compression-risk/omni-col-press-hr-smoke-2026-08-05.json`。服务器保留官方
`results.json`、索引、mask、构建配置、manifest、POSIX timing 和 profiler trace；摘要
记录它们的 SHA-256，以便后续规模实验核对。
