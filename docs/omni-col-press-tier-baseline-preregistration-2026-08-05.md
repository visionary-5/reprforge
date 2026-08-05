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

运行后应产生 `full/`、`hpool/`、`agc/` 三套官方 index/result，以及 `timing/` 和 `profiles/`。在这些文件实际生成并校验前，本分支的结论保持 `NO_GO_LOCAL_ARTIFACTS_MISSING`。

## 6. 对 ReprForge 的直接价值

OmniColPress 是一个合适的表示 DAG 物理基线，但贡献边界要说清楚：

- Full→H-Pool 是同一冻结模型下的参数无关压缩边；
- 页面→AGC 是需要专用训练权重和 attention 的另一条构建边；
- ReprForge 可以统一记录它们的输入、revision、向量数、bytes、构建时间与质量，却不能假设二者共享同一个已经物化的中间 embedding。

这组数据真正要回答的不是“压缩是否有效”——OmniColPress 已经研究过——而是不同构建边是否能作为 ReprForge 可验证、可缓存、可调度的物理表示，并在统一成本账本下形成新的质量--存储--构建等待前沿。
