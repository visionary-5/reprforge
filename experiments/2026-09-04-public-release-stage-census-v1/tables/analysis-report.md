# Public release stage census

## Question

How often does a public ColPali-family release change the visual prefix, and how
often does it change only the language-model suffix and the retrieval head?
This bounds how often a post-vision semantic cut can exist at all.

## Method

For 30 Hugging Face repositories we parsed only the safetensors headers
(tensor names and shapes) plus `adapter_config.json`, `preprocessor_config.json`
and `config.json`. No weights were downloaded. Each tensor is classified by the
module it wraps: vision tower, vision-to-language merger, decoder layers, or the
terminal retrieval projection. For LoRA checkpoints the classification applies
to the wrapped module path.

## Result

- 19 adapter checkpoints were parsed (official releases, third-party
  multilingual adapters, and the three Jina v4 task adapters).
- **0 of 19 contain any vision-tower or merger LoRA tensor.**
  Every parsed adapter changes only decoder layers and, where present, the
  terminal projection. This is the structural reason a post-vision cut exists.
- Consecutive official transitions:

| Family | Transition | Same base | Same processor file | Target touches vision | Deepest legal cut |
|---|---|---|---|---|---|
| ColPali | colpali → colpali-v1.1 | True | True | False | post-vision |
| ColPali | colpali-v1.1 → colpali-v1.2 | False | True | False | raw pages |
| ColPali | colpali-v1.2 → colpali-v1.3 | True | False | False | post-vision |
| ColQwen2 | colqwen2-v0.1 → colqwen2-v1.0 | True | True | False | post-vision |
| ColQwen2.5 | colqwen2.5-v0.1 → colqwen2.5-v0.2 | True | False | False | post-vision |
| ColSmol | colSmol-256M → colSmol-500M | False | True | False | raw pages |

- Cross-vendor sharing: `Metric-AI/colqwen2.5-3b-base` and
  `tsystems/colqwen2.5-3b-base` ship a first shard whose LFS sha256 and size
  (`6b45c7afe391…`, 4,997,750,760 bytes) are identical to
  `vidore/colqwen2.5-base`. That shard holds all 385 vision-tower tensors, the
  5 merger tensors and the first 238 decoder tensors. Three vendors' retrievers
  therefore share a bitwise-identical visual prefix; only their LoRA suffixes and
  randomly initialised `custom_text_proj` differ. Jina-embeddings-v4 goes further
  and ships three task adapters (1,512 decoder LoRA tensors, 6 terminal tensors)
  over one frozen Qwen2.5-VL backbone.

## Processor caveat that the paper must state

File identity of `preprocessor_config.json` is a poor proxy for prefix validity
in both directions:

- `vidore/colqwen2.5-v0.1` ships `max_pixels = 12,845,056`; `v0.2` and
  `vidore/colqwen2.5-base` ship `max_pixels = 602,112`. `colpali_engine` 0.3.12
  only overrides this budget when `max_num_visual_tokens` is passed, so the
  shipped v0.2 release *does* lower the page resolution. The paper pins the
  processor to the Qwen2.5-VL base contract (`f2058c71…`, 12,845,056 pixels),
  i.e. the v0.1 contract. Under the paper's own version tuple the shipped
  v0.1→v0.2 release is therefore an adapter **and** processor change; the
  post-vision cut is legal only because the operator keeps the processor
  component fixed. This must be stated in the paper, and it is a good
  illustration of why the processor is a first-class version component rather
  than a defect of the method.
- `vidore/colpali-v1.2` → `v1.3` changes the processor file hash but the only
  differing key is `_valid_processor_keys`, which does not affect outputs; the
  paper's collection-scoped output certificate is the right tool here.
- `vidore/colpali-v1.1` → `v1.2` switches the base from `mix` to `pt`, so the
  prefix is invalid even though the adapter itself has no vision tensors. The
  base fingerprint, not the adapter, decides.

## Claim boundary

The census covers public checkpoints on the Hub as of 2026-09-04. It says how
often a post-vision cut is *legal*; whether it is *worth it* is the codec and
cost question answered by the other experiments.

## Byte-level prefix identity (verified 2026-09-04, server-side)

SHA-256 of the raw bf16 bytes of two vision-tower tensors
(`visual.blocks.0.attn.qkv.weight`, `visual.patch_embed.proj.weight`),
range-fetched from the vendors' base shards and hashed locally:

| Checkpoint | qkv.weight | patch_embed | verdict |
|---|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | 72a8e6c1a0e2 | a68f46dbc6a0 | reference |
| local 3B base used in all experiments | 72a8e6c1a0e2 | a68f46dbc6a0 | identical |
| vidore/colqwen2.5-base | 72a8e6c1a0e2 | a68f46dbc6a0 | identical |
| Metric-AI / tsystems 3B bases | shard 1 LFS digest identical to vidore's | | identical |
| Qwen/Qwen2.5-VL-7B-Instruct | d9e147b82330 | 445267f7f856 | reference (7B) |
| nomic-ai/colqwen2.5-7B-base | ec9f8c157ddc | 4db939ec8db1 | **different vision tower** |

Consequence: the 3B ecosystem (vidore, Metric-AI, T-Systems, Nomic-3B) shares
one visual prefix byte for byte, so one stored cut is legal for all of them. The
Nomic 7B base does not share Qwen's 7B vision tower; a cut emitted from
Qwen2.5-VL-7B-Instruct is illegal for it and the planner must route raw. This is
a real instance of the census rule that the base fingerprint, not the adapter,
decides legality.
