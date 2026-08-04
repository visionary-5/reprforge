# Causal Hard Frontier 预注册

日期：2026-08-05

分支：`exp/causal-hard-frontier`

基线：`e05fe87`（包含 hard-fair selection-first provenance 修正）

状态：在实现独立 policy、运行等价性或读取 transfer quality 前冻结。

## 研究问题

`B=32` hard-constrained oracle 已在 HR 选择并冻结，且迁移到 Finance 后通过注册 GO；事后
审计显示被选配置的 quality weight 为 0，调度只需要已到达 cohort 和当前索引状态。本轮
不再寻找新配置，而是回答更严格的问题：能否把同一规则实现为一个接口上无法接收 qrel
或未来 arrival array 的事件驱动在线 policy，并跨五个已有 ViDoRe 域复现收益？

若可以，它才是 paper 方法候选；若只在 oracle replay 内成立，上一轮仍只能称 headroom。

## 冻结方法与接口

独立 policy 名固定为 `hard_budget_frontier`，代码放在独立模块，不使用
`multiobjective_oracle` 名称。全部常数冻结：

- physical batch max 8，visible window 64，active LRU 80 pages；
- younger-bypass budget `B=32`；
- completion weight 0.75，deadline weight 0.25；
- deadline scale 256 unit-time，underfilled-batch timeout 16 unit-time；
- completion density denominator 为 exact next-query demand cost，epsilon 1；
- 相同 utility 时按更老 arrival rank、query id 排序；
- 不 prefetch，不服务尚未 arrival 的 query，不改变 candidate union。

policy 对外只允许以下事件或当前状态：

1. `arrival(query_id, locator_cohort, event_time)`；
2. `timer(event_time)`；
3. `dispatch(event_time)`，以及上一批服务后当前 compiled set / active LRU 更新。

构造器和方法签名不得接收 quality/qrel gain、完整 arrival order、完整 arrival time array、
下一 arrival 时间或内容、query 总数/end-of-dataset 标志。模拟 harness 可以持有冻结 trace
并逐事件喂入；post-hoc evaluator 可以在 batch 已选择后发布该 batch 的 quality gain，但
不能把 gain 回传给 policy。

## 事件语义

- pending 为空时，外部 event loop 等待 arrival；
- pending 达到 8 时立即 dispatch；否则以最老 pending arrival+16 设置 timer；
- timer 前发生 arrival，只把该 query/cohort 加入状态并重评是否满批，不查看后续事件；
- timer 到期仍有 pending 时 dispatch；
- 无显式 end-of-stream 信号，最后未满批也等到已经设置的 timer；
- 服务期间到达的 query 只在当前 atomic batch 完成后作为 arrival event 交给 policy；
- batch 每个 slot 在 virtual compiled/LRU 上重算 cost 和 utility，并逐 slot 更新 bypass。

## 第一硬门：与 B32 reference 完全等价

数据固定为本机已有 `data/current-anytime` 的 HR/Finance BM25 Top-20 traces；arrival 为
burst/Poisson，seeds `20260804`--`20260808`，共 2 domain × 2 arrival × 5 seed = 20 个
replay cells。

reference 是提交 `461d03c` 的 `multiobjective_oracle`，固定 B32 config。新 policy 必须在
每个 cell 逐项满足：

- dispatch order tuple/hash 完全相同；
- per-query completion/sojourn elapsed tuple 完全相同；
- charged work、build/reload/hit、final union 完全相同；
- bypass tuple、budget violations、batch publication trace 完全相同；
- 不读取 qrel/future-array 的接口测试通过。

任一不等价则判 `CAUSAL MATERIALIZATION FAIL`，仍可报告差异，但不能称部署候选。该门不
比较由 provenance metadata 导致的整个 JSON 文件 hash。

## 五域冻结 transfer

仍使用 BM25 Top-20 locator cohort 与相同 unit cost，不改任何常数。目标域固定为：

1. ViDoRe v3 HR；
2. ViDoRe v3 Finance-EN；
3. ViDoRe v3 Computer Science；
4. ViDoRe v3 Industrial；
5. ViDoRe v3 Pharmaceuticals。

HR/Finance 读取 `data/current-anytime` manifest pair；另外三域只读取本机已经存在的
`/private/tmp/reprforge-vidore-domain-matrix-v1/<domain>/` 三个冻结 NPZ。运行前校验记录的
SHA；不重新下载数据、不生成 embedding、不调用 GPU。qrel/visual scores 只用于 dispatch
后的 quality curve，scheduler 只见 BM25 cohort。

每个可用域运行 burst/Poisson × 5 seeds。若路径不存在或 SHA 不符，写
`unavailable`/`digest_mismatch`，不寻找替代数据。比较固定为：FIFO、frontier、
overlap-only、HR-selected bounded CaGR、`hard_budget_frontier`。

每个方法报告：

- elapsed unit-time、charged work、unique compiled pages 三轴 quality regret/AUC；
- mean/P50/P95/P99/max sojourn 与 wait；
- work/query、build/reload/hit、final union；
- younger-bypass P50/P95/P99/max、`>=64` fraction、B32 violation；
- batch 数/利用率、timer 触发次数与等待量；
- deterministic control-plane operation counts：utility evaluations、page probes、
  feasibility comparisons、timer/dispatch events、每 query operations。墙钟微基准不进入
  deterministic 主 JSON，也不包装成 GPU throughput。

每个 domain×arrival×seed×axis 的 common horizon 取五方法该 cell 最大终点。ViDoRe 没有
自然生产时间戳，burst/Poisson 是固定压力模型，不能写成真实流量。

## ColModernVBERT availability-bound transfer

只接受能在当前主机直接读取、同时含 per-query Top-20 cohort、corpus identity 和 post-hoc
quality gain 的完整 replay trace。已有 summary JSON、aggregate schedule 或远端路径指针
不算本地可回放 trace；本轮不下载、不复制远端大文件、不启用 GPU。

若合格 trace 可读，使用完全相同五方法和 arrival/seeds 运行；否则在 JSON 明确列出已知
artifact manifest/hash 与缺失组件，判 `not_run_missing_local_replay_trace`。Modern 缺失不
反向改变五域配置或 gate，也不能被表述为跨表示成功。

## Paper 主方法候选门

只有同时满足以下条件，结论才为 `PAPER METHOD CANDIDATE`：

1. 20/20 HR/Finance reference cells 通过 byte-equivalent 硬门；
2. 五个 ViDoRe 域至少 4 个可用，所有已运行 cell parity 且 B32 violation=0；
3. 对五域可用的 domain×arrival 聚合 cell，hard method 相对 bounded 的 mean sojourn、
   work/query，以及相对 frontier 的 elapsed regret，任一 ratio 不得超过 1.10；
4. 上述三 ratio 的跨 cell median 全部 `<=1.0`；
5. 至少 60% cell 在三个主轴至少一个改善 `>=5%`；
6. P99 sojourn 在至少 80% cell 不超过 `1.05×min(bounded, frontier)`；
7. 若本地 Modern trace 可用，其所有 cell 也必须 parity、零 B32 violation；若不可用，结论
   可以是 ViDoRe paper-method candidate，但必须显式标注“跨 retriever 未验证”。

不根据 transfer 结果改阈值、方法常数、arrival 或 comparator。即使通过，也只证明 replay
级可部署候选；真实并发、GPU cost estimator 校准和 joint age/slowdown tail constraint
仍是系统落地门。

## 固定产物

- 本合同独立提交；
- 独立 event-driven policy、接口/状态审计和 reference equivalence 测试；
- 20-cell 等价 JSON、五域 transfer JSON、Modern availability audit；
- 完整 Markdown 报告、全仓测试、连续两次 byte-identical JSON、clean local commit；
- 不推远程。
