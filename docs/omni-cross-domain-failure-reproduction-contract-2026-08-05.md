# OmniColPresseval 跨领域失败复现协议

日期：2026-08-05  
状态：在 Computer Science / Finance 结果产生前冻结  
目的：复现失败边界，不选择新策略、不调参

## 1. 要回答的问题

HR 已经显示：H-Pool 这类廉价表示既可能漏掉 Full 的高分页面，也可能已经包含正确页面、但自身排序能力不足。跨领域实验只回答三件事：

1. “候选没进来”和“候选进来但排错了”是否在不同领域都存在；
2. Full 对 H-Pool 候选重排能否稳定修复后一类错误；
3. AGC 是否稳定补足 H-Pool 的候选覆盖，以及这项收益是否值得第二模型和第二索引的成本。

本轮不训练 query policy，不按新领域结果修改阈值，也不继续搜索池化或压缩算子。

## 2. 数据集

| 领域 | 查询 | 页面 | 正例关系 | 用途 |
|---|---:|---:|---:|---|
| ViDoRe v3 HR | 318 | 1,110 | 既有完整评测 | 已观察的发现域 |
| ViDoRe v3 Computer Science | 215 | 1,360 | 1,049 | 第一复现域 |
| ViDoRe v3 Finance-EN | 309 | 2,942 | 1,461 | 第二复现域 |

Computer Science 与 Finance 都使用全部英文查询、全部页面和完整 qrels，不抽样。导出器按文件名排序读取全部 parquet 分片；Finance 的三份 corpus 分片不能只读取第一份。

原始分片 SHA-256 已固化在各自导出 manifest 中。服务器输入目录同时记录导出后 `corpus.jsonl`、`queries.jsonl`、`qrels.jsonl` 与 manifest 的 SHA-256。

## 3. 固定表示与检索路径

- **Full**：高保真多向量表示，也是本轮的 teacher 与最终重排器；
- **H-Pool**：同一 Full 模型产生的低成本 pooled 定位表示；
- **AGC**：独立 checkpoint 的另一种低成本定位表示；
- **Cascade-k**：只允许 Full 在 H-Pool Top-k 候选内重排，固定 `k ∈ {20, 50, 100}`；
- **Union-100**：H-Pool Top-100 与 AGC Top-100 的去重并集，仅作候选覆盖和成本诊断，不当作已部署方法。

模型 revision、上游代码 commit、兼容补丁、输入文件、GPU 型号和 Python 包版本必须进入运行 manifest。三个表示在同一个领域上必须使用同一批查询、页面和 qrels。

## 4. 固定指标与失败定义

### 4.1 Teacher containment（严格包含）

对每条查询，寻找最小候选深度，使 H-Pool 候选集合包含 Full Top-10 的全部页面。类别固定为 `20 / 50 / 100 / Full`。这里的 Full 表示 Top-100 仍不能完全包含，并不自动等价于“正确证据逃逸”。

### 4.2 Quality preservation（质量保持）

寻找最小路径，使该路径的 nDCG@10 不低于 `Full nDCG@10 - 0.001`。类别同样为 `20 / 50 / 100 / Full`。

### 4.3 两种 escape（逃逸）

- **ranking escape**：Full Top-10 页面不在 H-Pool Top-100；
- **teacher-evidence escape**：上述逃逸页面同时是 qrels 中的正例。

论文判断以第二种为主。第一种可能只是 Full 与 H-Pool 对非相关页面的排序差异。

### 4.4 修复与互补

- H-Pool harm：`H-Pool nDCG@10 < Full nDCG@10 - 0.001`；
- cascade recovery：Cascade-100 恢复到上述容差内；
- AGC recovery：teacher-evidence escape 页面进入 AGC Top-100；
- union gain：H-Pool/AGC Top-100 并集的 qrel recall@100 高于单独 H-Pool。

同时报告 H-Pool、AGC、Full 索引字节数。AGC 的第二 checkpoint 驻留、第二次 query encoding、融合去重延迟在未实测前必须列为未计成本。

## 5. 风险信号审计

只审计运行时可观察且不使用 qrels 的信号。AUROC 仅作 failure-boundary 诊断，不据此现场生成策略。如果某领域没有正例或没有负例，记为不可计算，不把它误写为 0.5。

跨领域汇总同时保留：

- 不依赖 AGC 的最佳信号；
- 依赖 AGC 的最佳信号；
- 每个领域的正例数量，防止用极少数事件夸大 AUROC。

## 6. 判读规则

实验结束后按以下顺序判断，不以单一平均 nDCG 下结论：

1. 若多个领域的大多数 H-Pool 质量损失都满足“候选已包含、Full 重排可恢复”，支持“表示分工 + 动态访问”，而不是继续优化统一压缩表示；
2. 若 teacher-evidence escape 在多个领域占主要损失，核心瓶颈转为多定位器候选生成；
3. 若 AGC 只带来很小或不稳定的证据覆盖收益，不能把双 locator 当作主方法；
4. 若廉价、无需 AGC 的风险信号跨领域失效，则 query-aware compiler 仍有 oracle headroom，但在线策略尚未解决；
5. 任一结论都必须同时报告质量、候选覆盖、索引空间和未计生命周期成本。

本协议之后允许修复实现错误，但不得在看到 Computer Science / Finance 结果后修改上述类别、深度、容差或主判读口径。
