# ReprForge / RBRC v0 冻结协议

日期：2026-08-06  
状态：**在任何 RBRC 方法级未见领域结果打开之前冻结**  
机器配置：`configs/rbrc-v0-frozen-2026-08-06.json`

## 1. 这轮究竟验证什么

RBRC v0 不再问“哪一种压缩表示最好”，而是问：

> 已有廉价定位表示和高保真页面表示时，系统能否先用历史工作负载排除质量风险过高的
> 物理计划，再根据当前高保真页面的驻留状态选择访问深度，在无未来查询信息的条件下
> 减少冷读取；证据不足时能否明确拒绝优化，而不是静默牺牲质量？

因此 v0 的方法对象是**查询计划程序**，不是 H-Pool、BM25 或 ColPali 本身。H-Pool 和
BM25 提供候选页面，Full/ColPali 提供候选内高保真排序，RBRC 决定本次需要访问多少个
高保真页面。

论文中只使用 `empirically calibrated`（经验校准通过）或 `calibrated-safe`，不使用
“逐查询严格安全”。有限样本上界只能支持工作负载分布上的经验风险声明。

## 2. 表示栈和数据角色

两条表示栈分别编译，不共享证书。

| 表示栈 | 廉价定位器 | 高保真表示 | 校准域 | 未见域角色 |
|---|---|---|---|---|
| BM25→ColPali | BM25 | ColPali-v1.1 | HR、Finance EN | IRPAPERS，仅为 RBRC 方法未见；项目早期已打开，不能称全局 sealed |
| H-Pool→Full | OmniColPress H-Pool | 同 checkpoint 的 Full ColBERT | Pharmaceuticals、Industrial | Energy FR、Physics FR，最终新领域转移 |

最终域和版本已经写死：Energy 使用
`0f6e77a3f73911e7c7834835391229d2a623e8c0`，Physics 使用
`c5a4712eeeaf5194c918466ebc20c137b6c82c35`；查询语言固定为 French。Full/H-Pool
checkpoint 固定为 `hltcoe/ColBERT_qwen2.5-vl_colpali` 的
`14a7bb3328187705ff153e3511a47f9abb144054`。

未见域只允许在本协议和评测器提交之后运行。出现失败时保留失败结果；任何阈值、计划
或缓存规则的改变都必须使用新的 protocol ID，不能覆盖 v0。

## 3. 候选计划空间

高保真参考计划固定为 `Static(50)`。这里的 Full 指高保真**页面表示**，不是全语料逐页
评分；主问题的强基线从一开始就是 locator Top-50 后访问 50 个高保真页面。

BM25→ColPali 的原子深度为 `{20, 30, 40, 50}`；H-Pool→Full 为 `{20, 50}`。对每个低于
50 的 floor `k` 枚举两类程序：

1. `Static(k)`：所有查询固定访问 Top-k；
2. `Guard(k, 50, B)`：若 Top-50 当前需要的冷页不超过预算 `B`，访问 Top-50；否则
   访问 Top-k；若 Top-k 也不满足预算，则 abstain 并执行 Top-50，完整计费。

在线状态 `s_t` 是当前 resident page set。逻辑成本固定为

\[
C(P\mid s_t)=\sum_{x\in P\setminus s_t}1.
\]

缓存容量为语料页面数的 20%，冷访问预算 `B=40`。嵌套计划中，预算可行时优先更深的
高保真计划，而不是总选择成本最低的 Top-k。执行过的高保真页面全部准入缓存，主实验
统一使用 LRU；固定 Top-50 的 LFU、等页大小/成本 GDSF 和 Belady 作为强对照。

## 4. 质量风险和证书

每个查询的 signed regret 定义为

\[
L(q)=\operatorname{nDCG@10}_{Top50}(q)-
\operatorname{nDCG@10}_{candidate}(q).
\]

正数表示候选方法变差。事前固定两种尺度：

- 工作负载平均等价容忍度 `epsilon_mean = 0.002`，即 0.2 个 nDCG 百分点；
- 单查询质量违规阈值 `epsilon_query = 0.05`；
- 允许的经验违规率 `delta = 0.05`；
- 违规率 Wilson 95% 上界不得超过 0.10。

每个校准域使用原始顺序和 20 个固定随机排列。由于同一查询在不同排列中重复出现，不能
把所有 query-order pair 假装成独立样本。证书先对每个查询在 21 个顺序上的损失取平均，
然后以查询为独立单位做 4,000 次 bootstrap。一个程序必须在**每个校准域**同时满足：

1. 平均损失的单侧 95% bootstrap 上界不超过 0.002；
2. 21 个顺序的平均损失 p95 不超过 0.002；
3. 顺序平均后的单查询损失超过 0.05 的比例不超过 5%；
4. 该比例的 Wilson 95% 上界不超过 10%。

在通过的程序中，离线编译器选择校准阶段平均冷读取最低者。Static 和 Guard 分开编译，
用于核心消融。若某一类没有非参考程序通过，该类输出 `ABSTAIN/FIXED_TOP50`，而不是
放宽阈值。

`epsilon_mean=0.002` 是本协议的正式边界。早期文档把 HR 的 -0.001595 按探索性
0.001 边界标成 NO-GO；正式论文不得同时沿用两个口径，应说明探索阈值已由本次事前冻结
替代。

## 5. 在线算法

```text
offline_compile(calibration_workloads):
    enumerate Static(k) and Guard(k, 50, B)
    replay natural order plus 20 frozen permutations
    compute finite-sample quality gates per calibration domain
    S_static <- all certified Static programs
    S_guard  <- all certified Guard programs
    return lowest-calibration-cost member of each set, or ABSTAIN

online_guard(query q, resident set R, compiled floor k):
    candidates <- cheap_locator(q, Top-50)
    if cold(candidates[:50], R) <= 40:
        plan <- Top-50
    else if cold(candidates[:k], R) <= 40:
        plan <- Top-k
    else:
        plan <- ABSTAIN/FIXED_TOP50
    execute plan; admit accessed high-fidelity pages; update LRU state
```

这不是 query classifier。查询内容不直接预测难度，质量许可来自离线工作负载，运行时
决策只使用候选页、驻留状态和预算。

## 6. 四组核心消融

所有消融使用相同候选流、20% 容量和 LRU，避免把缓存算法收益混入查询规划器。

| 方法 | 安全编译 | 驻留感知 | Abstain | 执行规则 |
|---|---:|---:|---:|---|
| 固定 Top-50 | 否 | 否 | 否 | 永远 Top-50 |
| 仅安全编译 | 是 | 否 | 是 | 最便宜的认证 Static；无则 Top-50 |
| 仅驻留感知 | 否 | 是 | 否 | 最低 floor 的 Guard，不做质量认证 |
| 完整 RBRC | 是 | 是 | 是 | 最便宜的认证 Guard；无则 Top-50 |

BM25→ColPali 开发域编译的冻结前结果是：所有 Static 均被拒绝；Guard-20 在 HR 被
拒绝；Guard-30 在 Finance 被拒绝；只有 Guard-40 同时通过两个域。这个结果来自允许
参与方法形成的校准域，不是论文测试数字。它预示四组消融应形成的机制关系：仅安全
编译可能完全 abstain，纯驻留可能更省但违规，完整方法保留一个较保守的动态计划。

## 7. 未见域评测和硬 gate

每个未见域按原始顺序加 50 个固定随机排列运行，统一报告：

- 平均 nDCG@10 差、平均 signed regret；
- 损失超过 0.05 的查询比例；
- 最差单查询损失和 worst-5% CVaR；
- abstain 比例；
- 每查询逻辑冷读取；
- 相对同缓存固定 Top-50、固定 Top-50+LFU/GDSF 的收益；
- 同一实际请求流与 Belady oracle 的差距；
- 上述顺序级指标的均值、标准差、p05、p95、最小值和最大值。

单域通过要求：完整方法平均损失不超过 0.002，平均违规率不超过 5%；相对相同 LRU 的
固定 Top-50 至少减少 10% 冷读取；至少优于 LFU/GDSF 中一个；abstain 不超过 50%；
至少 90% 查询顺序减少冷读取。Energy 和 Physics 必须都通过，才允许进入论文主张的
GPU 物理成本阶段。IRPAPERS 只提供较早的失效预警，不能替代两个 sealed 域。

## 8. GPU gate 之后才做什么

本提交不启动 GPU，不下载大模型，也不报告延迟。两个 sealed 域逻辑 gate 通过后，才把
逻辑 miss 映射到以下真实链路：

```text
cold mmap / NVMe read -> host buffer -> H2D -> MaxSim -> end-to-end
```

每段报告 mean、p95、p99，并对缓存容量和冷预算做二维小 sweep。物理基线为固定
Top-50+LFU/GDSF、无质量约束驻留贪心、EdgeRAG/CaGR 风格缓存或预取，以及 Belady
上界；同时单列 abstain 回退的尾延迟尖峰。逻辑页下降在真实 I/O/端到端延迟中不能转化
为稳定收益时，系统 claim 判失败。

## 9. 可审计执行顺序

1. 提交本协议、配置、编译器、评测器、测试和 BM25 校准证书；
2. 记录提交哈希，之后才允许读取 IRPAPERS 方法未见结果；
3. H-Pool 校准证书只读取 Pharma/Industrial；
4. 下载 Energy/Physics 固定版本时先生成不含 qrel 的候选/分数面和哈希；
5. 固定证书与分数面后才加载 qrel，原样运行评测器；
6. 两域通过后再启动 GPU 物理链路计时。

评测器会校验配置的规范化 SHA-256、证书所属 protocol/profile 和输入哈希。机器结果中
同时保存证书哈希，使“先冻结、后打开”可以由 Git 历史和制品依赖共同审计。

