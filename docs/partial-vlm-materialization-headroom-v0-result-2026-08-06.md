# 部分 VLM 索引物化 headroom v0：子索引融合失败

日期：2026-08-06  
冻结提交：`ece018a`  
判定：**NO-GO；保留结果并将“选择空间”和“分数校准”拆开重测**

## 结果

v0 使用全页面 BM25 Top-100 与已物化视觉子索引 Top-100 做 RRF-60。三个已打开域的
完整混合索引相对 text-only nDCG@10 分别为：HR `+0.05922`、Finance `+0.00782`、
IRPAPERS `-0.03046`。但在 20%/40% 页面预算下，标签/Full-score oracle 也没有通过
冻结 gate，三个领域均判失败。

这不是足以否定部分物化的 oracle：v0 按**视觉子索引内部名次**计算 RRF。只物化少量
页面时，任意被选页面都可能从完整索引中的低名次变成子索引 Top-1，从而在所有查询中
获得过高视觉贡献。标签 oracle 选择的页面虽对某些查询相关，却会在其他查询中成为强
噪声。随机 5%/10% 页面甚至会把 nDCG 降到 0.17--0.29，说明评测首先测到了不可比较
分数，而不是页面选择 headroom。

selected-set z-score 的敏感性结果则显示明显但不可部署的空间：标签 oracle 实际只选择
HR 13.46%、Finance 3.73%、IRPAPERS 0.29% 页面时，nDCG@10 分别达到 0.6221、
0.6355、0.7467，均超过对应完整 z-score hybrid。Full-score oracle 在 IRPAPERS 20%
预算恢复约 97% 完整 hybrid 增益，在 HR/Finance 40% 预算分别恢复约 67%/77%。这些值
使用被禁止的 Full 分数，只说明值得进行干净分解。

## 在线 proxy

BM25 Top-20 页面事件存在复用：持久唯一构建相对非持久事件减少 HR 85.9%、Finance
70.0%、IRPAPERS 72.6%。但累计 materialization 最终覆盖 HR 80.6%、Finance 63.1%、
IRPAPERS 30.6%，且朴素持久 RRF 质量并不稳定。因此“有复用”不等于持久视觉检索索引
有价值；还必须同时证明覆盖比例、质量和相对 DVI 式非持久处理的摊销收益。

## 下一步

v1 不覆盖本结果。它只在 oracle 层固定页面在**完整视觉索引中的全局名次/全局分数
标定**，从而回答“若融合可比较，页面选择是否有空间”；同时继续报告 selected-set
RRF/z-score 作为可部署校准缺口。全局 rank/标定依赖未物化页面的 Full 分数，明确禁止
进入最终方法。

机器结果：
`results/compiler-feasibility/partial-vlm-materialization-headroom-v0-2026-08-06.json`。

