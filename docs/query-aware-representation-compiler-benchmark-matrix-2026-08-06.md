# Query-aware Representation Compiler：Benchmark 与实验矩阵

日期：2026-08-06  
阶段：先扩充可复现实证，再设计动态策略  
主问题：多模态检索系统应如何管理不同构建成本、驻留成本、访问成本和检索能力的表示

## 1. 实验原则

当前不继续搜索新的压缩算子，也不在两三个领域上反复调启发式规则。先固定表示和路径，扩大领域、检索器、负载和物理成本覆盖，再从稳定失败边界中定义 compiler。

每个实验必须同时回答四个问题：

1. **质量**：最终检索质量是否接近或超过高保真路径；
2. **覆盖**：正确证据是否进入候选集，还是仅在候选内排错；
3. **成本**：构建、存储、模型驻留、查询编码、索引读取和重排分别花了多少；
4. **适用边界**：哪些查询、领域和系统状态适合某条表示路径。

不允许只用平均 nDCG 支持动态系统，也不允许只用索引字节数代表总成本。

## 2. Benchmark 梯级

### Tier A：ViDoRe v3，多领域主矩阵

ViDoRe v3 是当前主 benchmark。它支持多阶段 pipeline，并提供页面相关性、定位和回答标注；官方 pipeline 框架也允许记录额外成本信息。

| 领域 | 查询 | 页面 | 当前状态 | 研究价值 |
|---|---:|---:|---|---|
| HR | 318 | 1,110 | Full/H-Pool/AGC/Cascade 完成 | 发现域，候选逃逸相对更多 |
| Computer Science | 215 | 1,360 | 完成 | 排序修正占主导，0 真实证据逃逸 |
| Finance-EN | 309 | 2,942 | 冻结队列等待 GPU | 第三域与表格密集场景 |
| Pharmaceuticals | 364 | 2,313 | 本地导出并上传完成 | 医药图表、长尾证据、多正例 |
| Industrial | 283 | 5,244 | 本地导出完成，上传中 | 最大且更难的技术图纸/图表域 |

随后按资源增加 Energy、Physics 和 Finance-FR。法语域用于检查 query encoder 与 locator disagreement 是否跨语言稳定，不参与英语策略选择。

### Tier B：REAL-MM-RAG，第二 benchmark 家族

优先子集：FinReport、FinSlides、TechReport、TechSlides。它强调真实 RAG 查询、表格密集文档和查询改写难度，适合验证：

- 同一信息需求在不同措辞下是否选择相同表示路径；
- 文本、表格和视觉证据对 locator 的需求是否不同；
- ViDoRe 上观察到的候选内排序规律是否迁移到另一套数据构造与标注体系。

接入前先做格式和许可审计，再决定是完整运行 Omni 表示，还是先用已有检索输出做路径模拟。

### Tier C：规模与长文档压力

候选为 ViMDOC/HEAVEN 公开数据与其他多文档长文档集合。该层不承担主质量结论，主要测：

- corpus 扩大后 Full mmap、候选物化和索引驻留是否仍可行；
- 固定 Top-100 是否仍是合理候选预算；
- single-vector locator + multi-vector reranker 与 H-Pool + Full 的差异。

### Tier D：端到端生成，只作外部效度

M²RAG、Visual-RAG、ViDoSeek 等回答生成 benchmark 放在检索路径稳定之后。它们回答“检索差异是否影响最终回答”，不能替代当前 page-level failure analysis。

## 3. 固定表示和访问路径

每个 Tier A/B 领域至少保留以下路径：

1. Full 全库检索；
2. H-Pool 全库检索；
3. AGC 全库检索；
4. H-Pool Top-20 → Full；
5. H-Pool Top-50 → Full；
6. H-Pool Top-100 → Full；
7. H-Pool/AGC Top-100 候选并集，仅作多定位器诊断；
8. Full oracle 最小访问深度，仅作 compiler headroom，不作为部署结果。

后续新增路径必须解释自己替代了哪条现有路径，不能只在矩阵中追加一个更复杂组合。

## 4. 强 baseline

### 4.1 质量 baseline

- BM25 或文本稀疏检索；
- ModernVBERT 等单向量文本检索；
- ColQwen2.5-v0.2 等标准视觉 late-interaction 检索器；
- Nemotron ColEmbed V2 等当前强视觉检索器；
- Full teacher、AGC 和官方 ViDoRe v3 pipeline leaderboard 结果。

“Full”只表示当前 Omni checkpoint 的高保真状态，不得写成全局 SOTA 或绝对上界。Computer Science 上 AGC 的 nDCG@10 已高于 Full，证明不同 checkpoint 也属于能力异构的一部分。

### 4.2 多阶段系统 baseline

- 固定 Top-20/50/100 cascade；
- single-vector locator + multi-vector reranker；
- HEAVEN 类单向量初筛、多向量精排；
- Visual RAG Toolkit 类 pooling + multi-stage search；
- 所有查询 Full；
- 所有查询最廉价 locator；
- oracle 最小充分路径。

### 4.3 动态策略 baseline

在跨领域矩阵完成前不训练最终策略，但预留以下公平对照：

- 固定深度；
- 只按查询长度或分数 margin；
- 只按 locator disagreement；
- 只按系统状态，例如 Full 已驻留/未驻留；
- 查询信号 + 系统状态联合策略；
- 质量约束下的离线 oracle。

## 5. 指标矩阵

### 5.1 检索质量

- nDCG@1/5/10/100；
- Recall@1/5/10/100；
- 相对 Full 的逐查询增益/损失分布；
- 有害查询比例、P95/P99 质量损失；
- 达到 `Full nDCG@10 - 0.001` 所需的最小路径。

### 5.2 候选覆盖与排序修正

- qrel recall@20/50/100；
- Full Top-10 containment depth；
- ranking escape 与 teacher-evidence escape；
- 候选已包含但 locator 排错的查询数；
- Cascade 修复率；
- 多 locator 并集新增的正例、非正例和候选数量。

### 5.3 生命周期成本

- 每种表示的 build wall time、GPU time 与 peak memory；
- 索引字节、mask/metadata 字节和 checkpoint 字节；
- query encoding、locator search、Full rerank 分段延迟；
- 冷启动、热缓存和受限缓存下的 P50/P95；
- 构建失败、版本更新或文档修改时的失效与重建工作；
- 表示同时驻留、切换和回收的额外成本。

### 5.4 Compiler 指标

- 在给定质量约束下的平均访问字节/向量数；
- 每条查询的表示激活数量；
- Full fallback 率和多 locator 激活率；
- 相对固定 cascade 的质量—成本 Pareto frontier；
- 风险校准：预测失败概率与实际质量损失；
- 跨领域最坏退化，而不只是 pooled mean。

## 6. 已有关键数据

HR + Computer Science 共 533 条查询：

- H-Pool 有害查询 186 条，Cascade-100 修复 185 条；
- 181/186 的有害查询已经包含 Full Top-10 全部候选；
- 只有 1 条查询发生真实 teacher-evidence escape；
- 多 locator 并集把加权 qrel recall@100 从 0.93965 提高到 0.95949；
- 但 HR 的并集收益明显大于 CS，说明第二 locator 不应默认激活；
- CS H-Pool 索引仅为 Full 的约 1/21.7，但构建时间是 Full 的 3.52 倍。

这些结果已足以支持“生命周期问题成立”，但还不足以支持任何具体动态 policy 已经解决。

## 7. 运行顺序与资源约束

1. Finance：保持当前冻结队列，不修改协议；
2. Finance 完成后立即拉取小型 ranking/metrics，生成 HR/CS/Finance 汇总；
3. 根据 Finance 实测磁盘决定是否直接跑 Pharmaceuticals；
4. Industrial 的 Full 预计占用约 59 GB，仅在有明确空间或可验证的低精度存储路径后运行；
5. Tier A 至少四域完成后，再接 REAL-MM-RAG；
6. 先跑固定策略和 oracle headroom，再做可学习动态策略。

服务器当前空间不能无界保留所有 Full 索引。任何清理必须在 ranking、metrics、manifest、timing、索引字节和必要 query artifact 已复制并校验后进行；未经确认不删除原始实验产物。

## 8. 进入方法优化的门槛

满足以下条件后才进入 query-aware policy 设计：

1. 至少四个 ViDoRe v3 领域完成相同表示矩阵；
2. 至少一个第二 benchmark 家族完成格式与小规模可行性验证；
3. 候选逃逸与候选内排序失败在各域的比例已确定；
4. 固定 Top-k、双 locator 和 Full 的真实成本均已计量；
5. 找到至少一个跨领域可观测风险信号，或明确证明现有廉价信号不足。

如果最终廉价 query 信号仍不稳定，compiler 的第一版应优先做“系统状态感知的表示选择”，例如利用已驻留表示、缓存和重建状态，而不是强行声称可以准确预测每条查询的语义难度。
