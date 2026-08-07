# DVI-like raw-page verifier GPU pilot v0：视觉定位在相同在线预算下修复文本逃逸

日期：2026-08-07  
协议冻结提交：`9bb2c84`，文本深度强基线补充提交：`ae6a753`  
判定：**GO，进入 10%/20% 物理部分视觉索引实验**

## 本轮回答的问题

本轮不声称完整复现或全面超过 Deferred Visual Ingestion（DVI）、AgenticOCR 或
EdgeRAG。它只回答一个更基础、也是部分视觉索引成立所必需的问题：如果查询时使用同一个
指令 VLM 检查原始页面，廉价文本定位器遗漏正确页面时，持久视觉检索能力能否用更少的
在线页面检查补回质量。

三个固定候选路由都使用相同的 Qwen2.5-VL-3B-Instruct 页面相关性 verifier：

- `BM25-20`：纯文本定位 20 页；
- `Visual-20`：完整持久视觉索引定位 20 页；
- `Hybrid-20`：BM25 10 页与视觉索引 10 页的并集。

verifier 只读取查询和原始页面图像，以 `logit(YES)-logit(NO)` 排序。Computer Science
仅用于校准门槛；Pharma 和 Industrial 使用确定性抽取的 100 个主查询盲测，未根据结果
修改 prompt、候选比例或模型设置。另报告所有 `BM25 Top-20 miss / Visual Top-20 hit`
查询作为逃逸压力集。

## 校准是否成立

| 领域 | 唯一 query-page pair | 正 pair | pair ROC-AUC | 每页均值 | 每页 p95 |
|---|---:|---:|---:|---:|---:|
| CS（校准） | 1,407 | 156 | 0.8625 | 150.9 ms | 155.3 ms |
| Pharma | 3,853 | 369 | 0.8453 | 157.3 ms | 164.5 ms |
| Industrial | 3,884 | 296 | 0.8329 | 153.9 ms | 161.9 ms |

冻结门槛要求 CS AUC 不低于 0.65 且至少有 20 个正 pair，实际显著通过。三域 AUC
稳定在 0.83--0.86，说明后续路由差异不是由一个失效的页面判断器制造出来的。

## 相同在线检查预算下的主结果

| 领域 | BM25-20 命中 | BM25-50 命中 | BM25-100 命中 | Hybrid-20 命中 | Visual-20 命中 |
|---|---:|---:|---:|---:|---:|
| CS | 97.5% | 100.0% | 100.0% | 97.5% | 100.0% |
| Pharma | 91.0% | 94.0% | 94.0% | **95.0%** | 95.0% |
| Industrial | 78.0% | 84.0% | **91.0%** | 87.0% | 90.0% |

`BM25-50/100` 是候选命中参考；DVI-like 系统若真正检查这些页面，需要分别支付最多
50/100 次 query-conditioned 原始页面处理。Hybrid 在 Pharma 用 20 页超过文本 100 页；
在 Industrial 用 20 页超过文本 50 页，并接近文本 100 页。

统一 verifier 重排后的 nDCG@10 为：

| 领域 | BM25-20 + VLM | Hybrid-20 + VLM | Visual-20 + VLM |
|---|---:|---:|---:|
| CS | 0.7121 | **0.7351** | 0.7250 |
| Pharma | 0.6274 | **0.6558** | 0.6373 |
| Industrial | 0.5136 | 0.5522 | **0.5557** |

Hybrid 在三个领域都高于纯文本；在 Pharma 超过完整视觉候选，在 Industrial 与完整视觉
只差 0.0035。完整视觉在 CS 经 verifier 后由 0.7349 降到 0.7250，说明通用 verifier
会引入排序噪声，更多视觉信息也不保证单调改善。这里保留该负结果，不把 Full visual 当作
严格 oracle。

## Candidate escape 是核心边界

| 领域 | 逃逸查询 | BM25-50 修复 | BM25-100 修复 | Hybrid-20 修复 | Visual-20 修复 |
|---|---:|---:|---:|---:|---:|
| CS | 6 | 66.7% | 100.0% | 100.0% | 100.0% |
| Pharma | 25 | 56.0% | 72.0% | **88.0%** | 100.0% |
| Industrial | 33 | 48.5% | **78.8%** | 72.7% | 100.0% |

Pharma 的 Hybrid-20 在压力集上重排后达到 0.4371 nDCG@10，Industrial 达到 0.2347；
BM25-20 因正确页面不在候选中始终为零。这个结果直接支持：查询时的 VLM 再强，也无法分析
没有被廉价索引定位到的页面。扩深文本候选有帮助，但会把每查询页面检查量从 20 提高到
50 或 100，且在 Pharma 仍不能覆盖视觉定位的长尾。

## 能说与不能说的结论

当前可以说：

1. DVI 的“廉价文本定位足够”存在稳定的跨域边界；
2. 文本和视觉定位器能力不等价，等候选预算融合能把集合优势转化为排序质量；
3. 存在值得进一步优化的构建成本与查询时页面检查成本工作区间。

当前仍不能说 ReprForge 已经优于 DVI、AgenticOCR 或 EdgeRAG。视觉候选仍来自已经构建的
完整视觉索引，本轮只证明能力 headroom，没有证明部分物化能够以 10%--20% 构建成本保留
这些候选。BM25 也不是最强的结构/OCR/dense-text 定位器；AgenticOCR 的区域级按需解析和
EdgeRAG 的同种 embedding 重生成具有不同问题边界。

## 下一步冻结 gate

下一轮物理构建 10% 和 20% 页面视觉索引，页面选择器不得读取测试 qrel 或完整视觉分数。
比较随机、文档均匀覆盖、静态视觉类型、廉价视觉风险与风险覆盖加 workload 收益策略，
并统一报告：

- 相对 text-only 到 full-hybrid 增益的恢复比例；
- candidate hit、nDCG@10、Recall@5/20；
- ingestion GPU 秒、索引字节、每查询原始页面检查数与 VLM 秒；
- BM25/OCR/metadata/dense-text Top-20/50/100 强基线；
- 随查询量增长的 DVI defer、ReprForge partial materialize 和 Full materialize 累计成本。

只有部分物理索引在至少两个领域以 10%--20% 构建成本保留大部分 Hybrid-20 收益，才进入
在线 promotion 和完整 defer--materialize phase diagram。否则本轮结果只作为 DVI 边界证据，
不包装成 ReprForge 方法贡献。

机器结果与原始 pair 分数位于
`results/compiler-feasibility/dvi-page-verifier-v0/`。
