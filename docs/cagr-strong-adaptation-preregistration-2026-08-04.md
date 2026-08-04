# Strong CaGR-style adaptation 预注册

日期：2026-08-04<br>
分支：`exp/cagr-strong-adaptation`<br>
状态：在运行本分支 HR 选择与 Finance 封存验证前冻结。

## 为什么需要第二道门

`cagr-faithful-v1` 忠实实现 [CaGR-RAG Algorithm 1](https://arxiv.org/html/2505.01164v1)，但把 IVF nprobe cluster set 映射成 BM25 Top-20 页面集后，`theta=0.5` 产生 98.4%–99.2% singleton groups。它足以排除“代码其实就是论文伪代码”，却可能被审稿人合理批评为没有为页面依赖图做最强适配。

本分支不修改 faithful 结果，而是构建**预先注册、只用 HR access graph 选参、Finance 完全封存**的 strongest CaGR-style counterfactual。它是压力测试，不是为 frontier 救结果。

## 数据隔离与禁止信息

- 开发域：HR，K=20，W64，request batch max=8，五个 permutation seeds `20260804`–`20260808`，同时包含 burst 与 Poisson。
- 封存域：Finance 的 access sets、arrival replay 和所有结果在 HR 配置冻结前不得读取或用于选择。
- 选择阶段只允许：已到达查询的 Top-20 页面 ID 集、Jaccard、arrival rank、compiled state、active LRU state、build/reload/hit/prefetch 计数。
- 选择阶段禁止：qrel、视觉分数、per-query quality gain、答案、Finance 指标。
- HR 与 Finance 的 qrel/视觉 surface 只在顺序冻结后做 post-hoc quality regret/AUC，不进入参数选择。

## 固定资源模型

所有候选沿用 `cagr-faithful-v1` 的两层状态与完整收费：

- `compiled` 页面持久存在、不淘汰；首次访问或预取支付 build=1 unit；
- `active_cache` 为 deterministic LRU；compiled 但不 active 时支付 reload=1 unit；hit=0；
- prefetch build/reload 全部进入 unit-cost；在首次需求前淘汰或流结束记 wasted/unused work；
- completion page-work 仍是查询原子发布时累计 unique compiled pages；
- 到达模型仍使用 page-work clock，不因选择结果改写；
- group-contiguous 执行不跨 group 填 request batch，固定组设计通过让 group size 接近 batch size恢复利用率，而不是隐藏跨组执行。

这不是墙钟模型。当前平台额度限制下不运行 GPU；报告必须保留 build/reload、cache hit、prefetch precision/waste 和 batch utilization。

## HR 候选网格

### Family A：lower-theta Algorithm 1

- `theta ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}`；
- `group_pool ∈ {20, 40, 64}`；
- `active_cache_capacity ∈ {40, 80, 160}`；
- membership 采用 faithful 主规则：当前 query 加入第一个与组内任一成员 Jaccard `>=theta` 的 group；
- group 按创建顺序、组内 arrival 顺序、组末预取下一组第一条 query。

共 54 个候选。论文 `theta=.5,pool64,capacity80` 必须在最终表中保留为 faithful 锚点。

### Family B：fixed-size Jaccard agglomerative

为避免阈值在不同 access-set 稀疏度下退化，加入一个固定组大小敏感性：

- `target_group_size ∈ {4, 8, 16}`；
- `group_pool ∈ {20, 40, 64}`；
- `active_cache_capacity ∈ {40, 80, 160}`。

每次对最老至多 `group_pool` 个 arrived/pending query 建组：

1. 在未分配 query 中，选择对其他未分配 query 的 Jaccard 总和最大者为 seed；tie-break 为 arrival rank；
2. 反复加入与当前 group **平均 Jaccard 最大**的 query，再以最大 single-link Jaccard、arrival rank tie-break，直到达到 target size 或 pool 为空；
3. group 按 seed 被选中的次序执行，组内按上述 agglomerative 加入次序执行；
4. group 末预取下一 group 第一条 query 的精确 Top-20；不跨 group 补 batch。

共 27 个候选。`target_group_size=8` 与当前 request batch 对齐；4/16 检查固定组规模敏感性。这一 family 是 CaGR-style 的强适配，不宣称是论文作者方法。

## 无监督 HR 选择规则

分别为 Family A、Family B 选择一个部署配置；另在全部 81 个候选中选 overall strongest。每个候选先聚合 HR 的 burst/Poisson × 五个 permutation。

Family A 的 `theta=0.50` 只作为 faithful 锚点保留在候选表中；名为 `hr_selected_lower_theta` 的部署配置严格只从 `theta<0.50` 的 45 个候选中选择，避免把 faithful 锚点重复命名成 lower-theta。overall strongest 仍可选择 `theta=0.50`。

令：

- `R_page` = 候选 mean completion pages / 同设置 FIFO mean completion pages；
- `R_cost` = 候选 mean completion unit-cost / 同设置 FIFO mean completion unit-cost。

选择分数固定为：

`score = mean_over_{burst,poisson}(0.5 * R_page + 0.5 * R_cost)`。

部署资格必须同时满足：

- 两种到达下 aggregate singleton fraction `<= 50%`；
- mean query-slot utilization `>= 50%`；
- final union parity；
- prefetch work 未被删除。

排序 tie-break 依次为：较小的两种到达 worst-case `max(R_page,R_cost)`、较小的 mean P95 unit-cost/FIFO、较大的 cache hit、family 名称与参数字典的 JSON 字典序。不得使用 quality、Finance 或结果后人为偏好。

如果 Family A 没有部署资格候选，明确报告 lower-theta NO-DEPLOYABLE，不从 Family B 借配置。Family B 和 overall 也同理。

## 封存评测方法

冻结以下三项后一次性在 Finance burst/Poisson 运行：

1. `hr_selected_lower_theta`；
2. `hr_selected_fixed_size`；
3. `hr_selected_overall`（可能与前两项之一相同）。

同表必须包含：

- `faithful_theta_0.5`；
- current causal overlap；
- FIFO、causal history popularity、offline static popularity；
- current frontier。

不实现 frontier+prefetch：frontier 没有预注册的稳定 group boundary，把预取加给它会同时改变被检验方法，不能用于这次反事实 gate。

## 强反事实判停门

对每个具有 HR 部署资格的、互不重复的 CaGR adaptation，在 Finance burst 与 Poisson 分别计算：

- `frontier_page_adv = 1 - frontier_mean_pages / adaptation_mean_pages`；
- `frontier_cost_adv = 1 - frontier_mean_unit_cost / adaptation_mean_unit_cost`。

只要出现以下任一情况，最终结论即 **STOP/DOWNGRADE “beat CaGR” claim**：

1. 任一 adaptation、任一 Finance arrival setting 的 `frontier_page_adv < 5%`；
2. 任一 adaptation、任一 Finance arrival setting 的 `frontier_cost_adv < 5%`；
3. adaptation 在 mean completion pages 与 mean unit-cost 上均不劣于 frontier，且至少一项严格更好（二维部署 Pareto dominance）。

quality regret、P95、starvation 作为完整解释报告，但不能挽救上述失败。相反，即使 frontier 两项都过 5%，若 P95、质量遗憾或 starvation 明显更差，必须限制论文表述，不得写无条件胜利。

只有所有部署型 CaGR adaptations 在 Finance 两种到达下均让 frontier 的 page 与 unit-cost 优势至少 5%，且没有 Pareto dominance，才是 **STRONG GO**。

## 必报指标与产物

- HR 81 候选的完整无监督 selection table、资格、score 与冻结配置；
- Finance 封存表及 gate；
- mean/P50/P95 completion pages 与 unit-cost；
- normalized quality regret、quality--work AUC；
- cache hit、demand/prefetch build/reload、prefetch precision/waste；
- singleton%、group mean/max、batch utilization；
- starvation fraction/max younger bypass；
- exact input/order/config hashes 与 final union parity；
- tests、JSON、Markdown report、干净 commit。

无论结果是否推翻 faithful GO，都不得修改本合同或删除负结果。
