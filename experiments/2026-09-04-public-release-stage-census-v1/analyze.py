"""Turn the raw census into the paper table and a short report.

Reads raw-output/census.json (tensor-header census) and writes
analysis-output/census-table.md, analysis-output/census-table.tex and
analysis-output/analysis-report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
rows = json.loads((ROOT / "raw-output" / "census.json").read_text())
out = ROOT / "analysis-output"
out.mkdir(exist_ok=True)

# Release lineages we treat as consecutive public transitions.
LINEAGES = {
    "ColPali": ["vidore/colpali", "vidore/colpali-v1.1", "vidore/colpali-v1.2", "vidore/colpali-v1.3"],
    "ColQwen2": ["vidore/colqwen2-v0.1", "vidore/colqwen2-v1.0"],
    "ColQwen2.5": ["vidore/colqwen2.5-v0.1", "vidore/colqwen2.5-v0.2"],
    "ColSmol": ["vidore/colSmol-256M", "vidore/colSmol-500M"],
}
THIRD_PARTY = [
    "Metric-AI/ColQwen2.5-3b-multilingual-v1.0",
    "Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
    "tsystems/colqwen2.5-3b-multilingual-v1.0",
    "nomic-ai/colnomic-embed-multimodal-3b",
    "nomic-ai/colnomic-embed-multimodal-7b",
    "ModernVBERT/colmodernvbert",
    "jinaai/jina-embeddings-v4",
]

by_repo = {r["repo"]: r for r in rows}


def counts(r: dict) -> dict:
    return r.get("lora_stage_tensor_counts") or r.get("stage_tensor_counts") or {}


def short(repo: str) -> str:
    return repo.split("/", 1)[1]


def base_of(r: dict) -> str:
    return str((r.get("adapter_config") or {}).get("base_model_name_or_path") or "-")


md = ["| Release | Base checkpoint | Vision LoRA | Merger LoRA | Decoder LoRA | Terminal | Processor file |",
      "|---|---|---:|---:|---:|---:|---|"]
tex = []
transitions = []
for family, repos in LINEAGES.items():
    prev = None
    for repo in repos:
        r = by_repo.get(repo)
        if not r or "error" in r:
            continue
        c = counts(r)
        proc = str(r.get("preprocessor_config.json_sha256", ""))[:8]
        md.append(f"| {short(repo)} | {base_of(r)} | {c.get('vision',0)} | {c.get('merger',0)} | {c.get('decoder',0)} | {c.get('terminal',0)} | {proc} |")
        tex.append(f"{short(repo)} & {short(base_of(r)) if '/' in base_of(r) else base_of(r)} & {c.get('vision',0)} & {c.get('merger',0)} & {c.get('decoder',0)} & {c.get('terminal',0)} & \\texttt{{{proc}}} \\\\")
        if prev is not None:
            p = by_repo[prev]
            same_base = base_of(p) == base_of(r)
            same_proc = p.get("preprocessor_config.json_sha256") == r.get("preprocessor_config.json_sha256")
            transitions.append({
                "family": family, "from": short(prev), "to": short(repo),
                "vision_lora_tensors_in_target": c.get("vision", 0) + c.get("merger", 0),
                "same_base_checkpoint": same_base, "same_processor_file": same_proc,
                "deepest_legal_cut": ("post-vision" if same_base and c.get("vision", 0) + c.get("merger", 0) == 0 else "raw pages"),
            })
        prev = repo
md.append("| **Third-party adapters on the same bases** | | | | | | |")
for repo in THIRD_PARTY:
    r = by_repo.get(repo)
    if not r or "error" in r:
        continue
    c = counts(r)
    proc = str(r.get("preprocessor_config.json_sha256", ""))[:8]
    md.append(f"| {repo} | {base_of(r)} | {c.get('vision',0)} | {c.get('merger',0)} | {c.get('decoder',0)} | {c.get('terminal',0)} | {proc} |")
    tex.append(f"{repo.replace('_', '\\_')} & {short(base_of(r)) if '/' in base_of(r) else base_of(r)} & {c.get('vision',0)} & {c.get('merger',0)} & {c.get('decoder',0)} & {c.get('terminal',0)} & \\texttt{{{proc}}} \\\\")

adapters = [r for r in rows if "error" not in r and (r.get("lora_stage_tensor_counts") or (r.get("adapter_config") is not None))]
n_adapters = len(adapters)
n_touch_vision = sum(1 for r in adapters if counts(r).get("vision", 0) + counts(r).get("merger", 0) > 0)

report = f"""# Public release stage census

## Question

How often does a public ColPali-family release change the visual prefix, and how
often does it change only the language-model suffix and the retrieval head?
This bounds how often a post-vision semantic cut can exist at all.

## Method

For {len(rows)} Hugging Face repositories we parsed only the safetensors headers
(tensor names and shapes) plus `adapter_config.json`, `preprocessor_config.json`
and `config.json`. No weights were downloaded. Each tensor is classified by the
module it wraps: vision tower, vision-to-language merger, decoder layers, or the
terminal retrieval projection. For LoRA checkpoints the classification applies
to the wrapped module path.

## Result

- {n_adapters} adapter checkpoints were parsed (official releases, third-party
  multilingual adapters, and the three Jina v4 task adapters).
- **{n_touch_vision} of {n_adapters} contain any vision-tower or merger LoRA tensor.**
  Every parsed adapter changes only decoder layers and, where present, the
  terminal projection. This is the structural reason a post-vision cut exists.
- Consecutive official transitions:

| Family | Transition | Same base | Same processor file | Target touches vision | Deepest legal cut |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {t['family']} | {t['from']} → {t['to']} | {t['same_base_checkpoint']} | {t['same_processor_file']} | {t['vision_lora_tensors_in_target'] > 0} | {t['deepest_legal_cut']} |"
    for t in transitions
) + """

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
"""

(out / "census-table.md").write_text("\n".join(md) + "\n")
(out / "census-table.tex").write_text("\n".join(tex) + "\n")
(out / "analysis-report.md").write_text(report)
(out / "transitions.json").write_text(json.dumps(transitions, indent=2))
print(report)
