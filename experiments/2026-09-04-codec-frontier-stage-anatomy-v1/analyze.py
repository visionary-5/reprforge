"""Summarise the codec frontier and stage anatomy run.

Usage: python analyze.py <output-root-with-result.json>
Writes analysis-output/{frontier-table.md,anatomy-table.md,frontier.tex,analysis-report.md,summary.json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "raw-output"
out = ROOT / "analysis-output"
out.mkdir(exist_ok=True)

summary = json.loads((src / "result.json").read_text())
results = summary["results"]
corpora = [c for c in ("arxivqa", "docvqa", "flickr") if c in results]
codec_order = ["bf16", "int8_token", "int4_group64", "pca512_int8", "pca256_bf16", "pca256_int8", "pca128_int8", "pca64_int8"]
LABEL = {"bf16": "exact BF16", "int8_token": "INT8/token", "int4_group64": "INT4/group64", "pca512_int8": "PCA-512 INT8",
         "pca256_bf16": "PCA-256 BF16", "pca256_int8": "PCA-256 INT8 (paper)", "pca128_int8": "PCA-128 INT8", "pca64_int8": "PCA-64 INT8",
         "stale": "stale (legacy index)", "procrustes_indomain": "Procrustes, in-domain", "affine_indomain": "affine ridge, in-domain"}


def f3(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------- frontier table
lines = ["| Route | " + " | ".join(f"{c} TA@10 / ΔnDCG@5 [95%] / IR÷terminal(fp32)" for c in corpora) + " |",
         "|---|" + "---|" * len(corpora)]
tex = []
frontier = {}
for m in ["stale", "procrustes_indomain", "affine_indomain", *codec_order]:
    cells = []
    for c in corpora:
        r = results[c]
        if m not in r["metrics"]:
            cells.append("–")
            continue
        ta = r["metrics"][m]["top10_overlap"]
        d = r["differences_vs_target_full"][m]["ndcg_at_5"]
        ci = r["paired_bootstrap_vs_target_full"][m]["ndcg_at_5"]["percentile_95"]
        ratio = r["payload_bytes_per_page"].get(m, float("nan")) / r["terminal_bytes_per_page"]["float32"] if m in r["payload_bytes_per_page"] else None
        cells.append(f"{f3(ta)} / {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]" + (f" / {ratio:.2f}×" if ratio is not None else ""))
        frontier.setdefault(m, {})[c] = {"ta10": ta, "dndcg5": d, "ci": ci, "ir_over_terminal_fp32": ratio,
                                          "payload_bytes_per_page": r["payload_bytes_per_page"].get(m)}
    lines.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    tex.append(f"{LABEL.get(m, m)} & " + " & ".join(cells).replace("×", "$\\times$").replace("–", "--") + " \\\\")

# ---------------------------------------------------------------- anatomy table
an = ["| Corpus | visual tok/page | preprocess s/page | prefix s/page | suffix s/page | leverage (pre+prefix)/total | replay PCA-256 s/page | measured saving |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
anatomy = {}
for c in corpora:
    r = results[c]
    sp = r["stage_seconds_per_page"]
    total = sp["preprocess"] + sp["prefix"] + sp["suffix_target"]
    replay = r["replay_seconds_per_page"].get("pca256_int8", float("nan"))
    saving = 1 - replay / total
    an.append(f"| {c} | {r['tokens']['visual_per_page']:.0f} | {sp['preprocess']:.3f} | {sp['prefix']:.3f} | {sp['suffix_target']:.3f} | {r['cut_leverage']:.3f} | {replay:.3f} | {saving*100:.1f}% |")
    anatomy[c] = {"visual_tokens_per_page": r["tokens"]["visual_per_page"], "stage_seconds_per_page": sp, "leverage": r["cut_leverage"],
                  "replay_pca256_s_per_page": replay, "measured_saving_vs_raw_encode": saving,
                  "pca_explained_variance": r.get("pca_explained_variance"), "hardware": r["hardware"]}

# ---------------------------------------------------------------- cross-backbone leverage
lev_dir = src.parent / "leverage-output" if (src.parent / "leverage-output").exists() else ROOT / "raw-output" / "leverage-output"
lev = ["| Backbone | vision params / total | corpus | doc vectors/page | preprocess s | vision s | forward s | vision share of forward | leverage |",
       "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
leverage = {}
for f in sorted(lev_dir.glob("*.json")) if lev_dir.exists() else []:
    d = json.loads(f.read_text())
    share = d["vision_params"] / d["total_params"]
    for c, a in d["corpora"].items():
        lev.append(f"| {d['family']} | {share*100:.0f}% | {c} | {a['doc_vectors']:.0f} | {a['preprocess']:.3f} | {a['vision']:.3f} | {a['total_forward']:.3f} | {a['vision_share_of_forward']:.2f} | {a['leverage']:.2f} |")
    leverage[d["family"]] = {"vision_param_share": share, "hardware": d["hardware"], "corpora": d["corpora"]}
leverage_md = "\n".join(lev) if len(lev) > 2 else "(leverage-output not present)"

report = f"""# Codec frontier and build-cost anatomy ({results[corpora[0]]['hardware']})

Holdouts, splits, adapters and processor are identical to the paper's Table 2 run
(`sigir-version-evolution-matrix-v1`); image batch size {results[corpora[0]]['scope']['image_batch_size']}.

## Frontier: fidelity vs retained bytes

{chr(10).join(lines)}

Reading guide: TA@10 is Top-10 agreement with the exact v0.2 index. The exact BF16
cut is the correctness check for the replay path (predicted 1.000). Every row
below it trades bytes for fidelity; the paper's PCA-256/INT8 is one point on
this curve, not the method.

## Anatomy: where raw build time goes

{chr(10).join(an)}

Leverage is the fraction of raw encode time spent before the language-model
suffix; no post-vision replay can save more than this. The measured saving is
replay (decode + suffix) against preprocess + prefix + suffix, excluding index
construction, which both routes share.

Flickr caveat: the paper's RTX 4090 table reports a 77.8% Flickr saving, but on
this A100 the same route saves 54.4% because a 226-token page is launch-latency
bound: the language-model suffix costs 0.056 s regardless of tokens, the same as
the whole visual prefix. At image batch 4 the suffix amortises to 0.020 s/page
and the saving is 76.5%, matching the 4090 (see extra-report.md). Small-image
savings depend on suffix batching, and the paper reports both regimes.

## Cross-backbone leverage (40 holdout pages per corpus, batch size 1)

{leverage_md}

A backbone whose vision tower is a small fraction of the forward pass (ColPali:
SigLIP-400M in front of Gemma-2B) leaves little for a post-vision cut to save;
this is the architectural reason the paper's ColPali route saved under 20% and
failed admission. ColSmol's leverage is high only because Idefics3 image
splitting is CPU-heavy; the cut saves preprocessing rather than GPU time.

## Explained variance of the collection basis

""" + "\n".join(f"- {c}: " + ", ".join(f"r={k}: {v:.3f}" for k, v in (results[c].get("pca_explained_variance") or {}).items()) for c in corpora) + """

## Claim boundary

One backbone, one release pair, one GPU. In-domain bridges are fitted on the
non-holdout queries of the same corpus, so they have an advantage the paper's
Energy-fitted bridges did not; if they still do not approach codec TA@10, the
compatibility-vs-target-semantics distinction does not depend on the fitting domain.
"""
(out / "frontier-table.md").write_text("\n".join(lines) + "\n")
(out / "anatomy-table.md").write_text("\n".join(an) + "\n")
(out / "frontier.tex").write_text("\n".join(tex) + "\n")
(out / "analysis-report.md").write_text(report)
(out / "leverage-table.md").write_text(leverage_md + "\n")
(out / "summary.json").write_text(json.dumps({"frontier": frontier, "anatomy": anatomy, "leverage": leverage}, indent=2))
print(report)
