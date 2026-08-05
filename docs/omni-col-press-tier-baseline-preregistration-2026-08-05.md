# OmniColPress 一级压缩基线：本地能力判定与 A100 预注册

> 日期：2026-08-05<br>
> 基线：`main@6d10439`<br>
> 当前判定：**本机不能产生真实 OmniColPress 结果；不生成合成结果。**

## 1. 先回答“现在能不能真实跑”

不能。阻塞点不是没有分数，而是没有压缩算法所需的原始表示。

本机 ReprForge 工作区共有 17 个 NPZ，其中 11 个保存完整 query-document 分数面；没有任何 NPZ 数组名为 `embeddings`、`query_vectors`、`doc_vectors` 或同义字段。结果 JSON 中能找到 46 个 `vectors.npy` manifest 引用、27 个唯一 SHA-256，但实体 `vectors.npy` 数量为 0。已有结果记录的 A100 路径 `/data/ldf/reprforge/results/mmdocir-e2e/route-bank-v1/embedding-bank` 在本机也不存在。[`results/README.md`](../results/README.md)明确说明 raw datasets、score banks、embeddings 和 physical indexes 不提交。

所以现有文件能重放排名和成本账本，却不能重新聚类 token。仅有最终 MaxSim 分数无法反推出逐 token query/document 向量。

本机 Python 环境同时缺少 `torch`、`scipy`、`transformers`、`peft`、`faiss` 和 `qwen_vl_utils`。本轮没有安装重依赖、下载约 15 GB 的两个 checkpoint、SSH 或占用 GPU。

机器可读审计见 [`results/feasibility/omni-col-press-local-capability-2026-08-05.json`](../results/feasibility/omni-col-press-local-capability-2026-08-05.json)。

## 2. 官方制品是否足够

官方制品本身基本足够，问题是本地尚未物化。

- [OmniColPress 官方仓库](https://github.com/hanxiangqin/omni-col-press)提交 `4a559677bbc8a3ea0c10322a721b52bb70d382ec`，有 MIT LICENSE、建索引 CLI、检索评测 CLI，以及 JSON/CSV/Parquet/Hugging Face datasets 接口。
- Full/H-Pool 官方 checkpoint 是 [`hltcoe/ColBERT_qwen2.5-vl_colpali`](https://huggingface.co/hltcoe/ColBERT_qwen2.5-vl_colpali)，固定 revision `14a7bb3328187705ff153e3511a47f9abb144054`，公开、非 gated、MIT，约 7.52 GB。
- AGC 官方 checkpoint 是 [`hltcoe/AGC_qwen2.5-vl_colpali`](https://huggingface.co/hltcoe/AGC_qwen2.5-vl_colpali)，固定 revision `14ba8fb11de7d15d5a87c7fa17e893bffcdd9020`，公开、非 gated、MIT，约 7.52 GB。
- 官方源码 `python -m compileall` 通过。但仓库没有 requirements/lockfile 或测试；README 安装段也没有显式列出源码直接导入的 `scipy`、`datasets`、`jsonlines`、`accelerate` 和 `safetensors`，所以 wrapper 会先做依赖 preflight，而不把“有 CLI”误写成“任意环境开箱即跑”。

## 3. 必须拆成两个实验口径

### 3.1 同一冻结表示下的纯压缩比较

这个口径只能比较：

```text
Full ColBERT  vs.  H-Pool
```

两者使用同一个 `ColBERT_qwen2.5-vl_colpali` checkpoint。官方模型卡明确说明只改变 `pooling="colbert"` 与 `pooling="hierarchical_clustering"`；查询不压缩。H-Pool 是参数无关的 Ward 层次聚类，因此可以归因于表示压缩。

### 3.2 端到端方法比较

这个口径可以比较：

```text
Full ColBERT  vs.  H-Pool  vs.  AGC
```

但 AGC 使用另一个训练后 checkpoint。它把 64 个学到的 Universal Query token 附加到文档序列，用这些 token 在最后一层到文档 token 的注意力生成 saliency，再选中心和加权聚类。[官方模型卡](https://huggingface.co/hltcoe/AGC_qwen2.5-vl_colpali)给出的默认输出为每页 64×2048。

因此 AGC 不能对现有 ColPali v1.1 128 维 bank 做忠实离线后处理，也不能从分数矩阵恢复。Full/H-Pool 与 AGC 的比较是“完整方法在同一页面、查询、qrel 上的效果/成本比较”，不是“同一冻结 query/doc embeddings 下只换压缩算子”。论文中报告的 60.0、56.4、56.7 等数字只能作为 prereg sanity range，绝不进入本地结果字段。

## 4. 冻结实验协议

最小 smoke 固定为最多 32 页、16 个查询、1 张 A100。所有方法使用同一份页面 JSONL、query JSONL、qrels 和图片目录；随机性、页面顺序和 query 顺序均不改变。

质量指标：

- nDCG@5、nDCG@10；
- Recall@5、Recall@10；
- 每查询 paired 差值，扩展实验时再做 bootstrap 置信区间。

物理指标：

- `index.pt` 中真实 tensor shape、dtype、`numel × element_size`；
- `index.pt`/`masks.pt` 的实际文件字节；
- 每页真实向量数，不使用论文固定 token 数代替本地测量；
- 建索引和 query evaluation 的 wall/user/sys 时间。

时间口径要特别谨慎：AGC 的 attention、选中心与聚类嵌在模型前向中，H-Pool 也混合了 GPU 相似度、CPU SciPy linkage 和 GPU 聚合。wrapper 会保存 POSIX time 与 PyTorch CPU/CUDA profiler trace，但在增加明确的 method-level instrumentation 前，**不得把端到端 build time 命名为“纯压缩 kernel 时间”**。

通过 smoke 的门槛：

1. 三个索引均由真实页面生成，官方评测能读回；
2. H-Pool 和 AGC 的每页向量数确为 64；
3. Full 与 H-Pool 使用完全相同 checkpoint revision；
4. 真实 bytes 来自 tensor/file，而不是官方 `get_stats()` 的估算值；该函数当前按 4 bytes/元素估计，不能代表 bf16 实际负载；
5. profiler 和计时文件齐全，无失败媒体替换记录；
6. smoke 通过后才扩展到现有可访问的 ViDoRe 域。

## 5. 唯一 A100 smoke 命令

wrapper 不覆盖已有输出目录，并核验 A100、官方源码 commit、依赖、输入行数和必要字段。它会在 A100 上下载两个固定 revision，而本轮没有执行。

```bash
CUDA_VISIBLE_DEVICES=<free-a100-id> OMNI_SOURCE=/path/to/omni-col-press OMNI_CORPUS_JSONL=/path/to/32-pages.jsonl OMNI_QUERY_JSONL=/path/to/16-queries.jsonl OMNI_QRELS=/path/to/qrels.jsonl OMNI_ASSETS=/path/to/images OMNI_MODEL_CACHE=/path/to/model-cache OMNI_OUTPUT_ROOT=/path/to/new-output bash experiments/omni-col-press-tier-baseline/run_a100_smoke.sh
```

共享服务器磁盘不足以无条件同时展开两个约 7.5 GB checkpoint 时，可以显式设置
`OMNI_CASES=full,hpool`，先运行共享同一 checkpoint 的受控压缩比较。该设置不会
下载 AGC checkpoint；第一阶段通过后，再用新的输出目录设置 `OMNI_CASES=agc`。
模型通过 `huggingface_hub.snapshot_download` 按固定 revision 下载，不依赖特定版本
的 `hf` 命令行入口。`OMNI_ATTN_IMPLEMENTATION` 默认为可移植的 `sdpa`，若使用
`flash_attention_2`，必须在结果清单中记录并保证所有方法口径一致。
Full/H-Pool 基座在启动 GPU 前还会校验 config、权重索引、tokenizer 和两个权重分片
的固定 SHA-256；仅存在 `config.json`、断点下载未完成或混入其他 revision 都不能
通过 preflight。
模型完整性通过后才创建输出目录，并写入 `run-manifest.json`：上游与 wrapper 提交、
两处兼容补丁及补丁后文件哈希、模型 revision、三个输入文件哈希、软件版本、GPU、
attention 实现和所请求方法。失败的模型准备不会留下看似有效的结果目录。

运行后应产生所请求方法的官方 index/result，以及 `timing/` 和 `profiles/`。在这些
文件实际生成并校验前，本分支的结论保持 `NO_GO_LOCAL_ARTIFACTS_MISSING`。

ViDoRe smoke 输入由 `tools/export_vidore_omni_smoke.py` 从冻结本地 Parquet 导出。
导出器只选择所有正相关页都能装入页预算的英文查询，写出源文件 SHA-256、查询和
页面 ID，并断言所选查询的 qrel 完整，避免为了 32 页上限静默删除金标准证据。
查询按“新增正相关页数、总正相关页数、query ID”的固定顺序贪心选择，因此 smoke
偏向证据页较少或可复用的查询；它只验证环境、张量形状和官方评价链路，绝不作为
有代表性的检索质量样本或论文效果数字。

官方提交会在选择普通 `multivec` 时也从构建器和评价加载器顶层导入可选
Fast-Plaid，而 Fast-Plaid 没有
与服务器 PyTorch 2.5 对应的公开 wheel。运行器因此应用并校验
`image_only_optional_fast_plaid*.patch`：只把两处 `FastPlaidIndex` 移到对应分支内
延迟导入，不修改 `MultiVecIndex`、模型、压缩或评价代码。结果 manifest 必须记录
官方 commit、两个补丁 SHA-256 和补丁后文件 SHA-256；这属于公开的环境兼容差异，
不能描述成上游源码零修改复现。官方当前源码还同时导入 Qwen3-VL processor，因此
运行器要求隔离环境中的 `transformers>=4.57.1`，不得改动已有 ViDoRe 固定环境。

## 6. 对 ReprForge 的直接价值

OmniColPress 是一个合适的表示 DAG 物理基线，但贡献边界要说清楚：

- Full→H-Pool 是同一冻结模型下的参数无关压缩边；
- 页面→AGC 是需要专用训练权重和 attention 的另一条构建边；
- ReprForge 可以统一记录它们的输入、revision、向量数、bytes、构建时间与质量，却不能假设二者共享同一个已经物化的中间 embedding。

这组数据真正要回答的不是“压缩是否有效”——OmniColPress 已经研究过——而是不同构建边是否能作为 ReprForge 可验证、可缓存、可调度的物理表示，并在统一成本账本下形成新的质量--存储--构建等待前沿。
