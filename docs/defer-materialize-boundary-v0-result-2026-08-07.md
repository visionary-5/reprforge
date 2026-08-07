# Defer–Materialize boundary v0：视觉能修复定位逃逸，但修复页不复用

日期：2026-08-07  
冻结提交：`1b0b1cc`  
判定：**DVI 定位假设存在边界；统一的 escape-triggered promotion gate 失败**

## 先澄清成本口径

持久页面级 retrieval embedding 不能替代针对新问题的 VLM reasoning。DVI 每次将页面和
问题交给 VLM 做定向分析；ReprForge 即使已持久化检索表示，最终回答仍可能需要同样的
query-conditioned VLM。可公平计入 ReprForge 收益的是：修复候选逃逸、改善正确页面次序、
减少需要交给 VLM 分析的候选页，以及摊销 retrieval embedding 的页面编码；不能把一次
retrieval encoding 当作以后所有 VLM QA 的替代品。

## Top-20 定位边界

| 领域 | 文本命中率 | 视觉命中率 | 文本漏失查询 | 视觉修复漏失 | 文本∪视觉命中率 |
|---|---:|---:|---:|---:|---:|
| HR | 90.25% | 92.45% | 31 | 19（61.3%） | 96.23% |
| Finance | 93.20% | 90.29% | 21 | 17（81.0%） | 98.71% |
| IRPAPERS | 92.78% | 83.89% | 13 | 6（46.2%） | 96.11% |

因此 DVI 的“廉价文本/结构定位足够”不能被当成普遍事实。完整视觉检索并非整体都比文本
强：Finance 和 IRPAPERS 的单独视觉命中率反而更低；但它能修复大量文本漏失，说明二者
是能力不等价的互补定位器。论文需要比较候选并集或校准融合，而不是把视觉索引当作文本
索引的替代品。

Top-100 时文本命中率已升至 HR 96.23%、Finance 99.68%、IRPAPERS 97.22%；视觉仍能
分别修复 10/12、1/1、3/5 个剩余漏失。DVI 可以用更深候选降低 escape，但会扩大需要
验证/分析的候选空间。当前 score surface 只给出乐观的“扫到首个相关页”页数，不能据此
伪造真实 VLM-QA 延迟。

## 修复证据是否复用

五折历史—未来切分中，Top-20 视觉修复页事件的历史复用率为：

| 领域 | 未来视觉修复查询 | 修复页事件历史复用率 |
|---|---:|---:|
| HR | 19 | 7.14% |
| Finance | 17 | 6.25% |
| IRPAPERS | 6 | 0.00% |

这与“所有相关页面”的复用率 HR 82.4%、Finance 62.4%、IRPAPERS 0% 完全不同：真正
发生 candidate escape 的页面主要是视觉长尾，而不是热门页面。协议要求至少两个领域的
修复事件复用率达到 20%，实际零个领域通过，因此不继续为统一的
`escape -> promote -> future reuse` 机制画累计成本 crossover。

## 对 design 的含义

结果支持将状态明确拆成两类，而不是用一个热度策略混合处理：

1. `S_cover`：主动覆盖低复用、文本不可见的视觉长尾。价值来自质量约束和防止 escape，
   不能依赖访问热度摊销；下一步要测 OCR 质量、页面视觉类型、布局和文档级代表性是否能
   在小预算下覆盖这些 repair pages。
2. `S_benefit`：针对可复用 workload 降低候选分析量或修正排序。它应学习历史中实际可
   观察的边际视觉收益，而不是把所有历史相关页都物化。

DVI 是低风险、低复用区域的自然最优端点；Full index 是视觉定位需求广泛且工作量足够大
时的端点。ReprForge 只有在“少量静态视觉风险覆盖 + 可摊销的收益页物化”同时优于扩大
DVI 候选深度与全量构建时才成立。

## 下一步 gate

先补齐页面级廉价视觉风险特征，并以本轮产生的 repair-page 标签做历史/未来隔离的覆盖
测试。必须同时报告 repair-page recall、页面预算、跨领域迁移和误物化比例。若 20%--40%
预算无法覆盖多数 repair pages，`S_cover` 不可实现，课题应停止或转向 DVI。只有覆盖
gate 通过，再构造分离后的二维 phase diagram，并在相同 query-time VLM 分析假设下比较
DVI、ReprForge 与 Full。

机器结果：
`results/compiler-feasibility/defer-materialize-boundary-v0-2026-08-07.json`。
