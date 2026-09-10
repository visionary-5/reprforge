"""Summarise the batch-size-4 control and the in-domain residual-MLP bridge.

Usage: python analyze_extra.py raw-output
Expects raw-output/output-batch4/result.json and raw-output/output-mlp/result.json.
Writes analysis-output/extra-report.md and extra-summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "raw-output"
out = ROOT / "analysis-output"
out.mkdir(exist_ok=True)
base = json.loads((src / "result.json").read_text())["results"]
# batched controls: (dir, corpus, batch label); DocVQA at batch 4 ran out of memory (4.4k tokens x 4 pages), so it uses batch 2.
CONTROLS = [("output-batch4", "arxivqa", "4"), ("output-batch2-docvqa", "docvqa", "2"), ("output-batch4-flickr", "flickr", "4")]
b4 = {}
for d, c, label in CONTROLS:
    f = src / d / c / "result.json"
    if f.exists():
        b4[c] = (label, json.loads(f.read_text()))
mlp = json.loads((src / "output-mlp" / "result.json").read_text())["results"]
corpora = [c for c in ("arxivqa", "docvqa", "flickr") if c in mlp]

lines = ["| Corpus | batch | preprocess s/page | prefix s/page | suffix s/page | leverage | replay PCA-256 s/page | saving | TA@10 bf16 | TA@10 pca256 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
summary = {"batch4": {}, "mlp_indomain": {}}
for c in corpora:
    controls = [("1", base[c])] + ([b4[c]] if c in b4 else [])
    for label, r in controls:
        sp = r["stage_seconds_per_page"]
        total = sp["preprocess"] + sp["prefix"] + sp["suffix_target"]
        rep = r["replay_seconds_per_page"]["pca256_int8"]
        lines.append(f"| {c} | {label} | {sp['preprocess']:.3f} | {sp['prefix']:.3f} | {sp['suffix_target']:.3f} | {r['cut_leverage']:.3f} | {rep:.3f} | {(1-rep/total)*100:.1f}% | {r['metrics']['bf16']['top10_overlap']:.3f} | {r['metrics']['pca256_int8']['top10_overlap']:.3f} |")
        if label != "1":
            summary["batch4"][c] = {"batch": label, "stage_seconds_per_page": sp, "leverage": r["cut_leverage"], "saving": 1 - rep / total,
                                    "ta10_bf16": r["metrics"]["bf16"]["top10_overlap"], "ta10_pca256": r["metrics"]["pca256_int8"]["top10_overlap"]}

m_lines = ["| Corpus | fit queries | epochs | val. token cosine before → after | stale TA@10 | MLP in-domain TA@10 | MLP ΔnDCG@5 [95%] | stale ΔnDCG@5 |",
           "|---|---:|---:|---|---:|---:|---|---:|"]
for c in corpora:
    r = mlp[c]
    ci = r["paired_bootstrap_vs_target_full"]["mlp_indomain"]["ndcg_at_5"]["percentile_95"]
    m_lines.append(f"| {c} | {r['fit_queries']} | {r['epochs']} | {r['validation_token_cosine_before']:.3f} → {r['validation_token_cosine_after']:.3f} | {r['metrics']['stale']['top10_overlap']:.3f} | {r['metrics']['mlp_indomain']['top10_overlap']:.3f} | {r['differences_vs_target_full']['mlp_indomain']['ndcg_at_5']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] | {r['differences_vs_target_full']['stale']['ndcg_at_5']:+.4f} |")
    summary["mlp_indomain"][c] = {"ta10": r["metrics"]["mlp_indomain"]["top10_overlap"], "stale_ta10": r["metrics"]["stale"]["top10_overlap"],
                                  "dndcg5": r["differences_vs_target_full"]["mlp_indomain"]["ndcg_at_5"], "ci": ci,
                                  "cos_before": r["validation_token_cosine_before"], "cos_after": r["validation_token_cosine_after"], "epochs": r["epochs"]}

report = f"""# Batch-size control and in-domain residual MLP

## Batch size 1 vs batched control (same A100, same holdouts; DocVQA batch 2 because batch 4 exceeds 80 GB)

{chr(10).join(lines)}

Batching amortises per-forward overhead. The question is whether it removes the
cut's advantage: if leverage at batch 4 stays close to batch 1 on document pages,
the paper's batch-1 timings do not flatter the cut. Small-image (Flickr)
leverage is expected to move most.

## In-domain Drift-Adapter residual MLP

{m_lines[0]}
{m_lines[1]}
{chr(10).join(m_lines[2:])}

Same recipe as the paper's Energy-fitted bridge (hidden 256, dropout 0.1, AdamW
3e-4, token batches of 256, early stopping on a 20% validation split), fitted on
the non-holdout 70% queries of each corpus. Token cosine rises as before; TA@10
does not move away from stale serving.
"""
(out / "extra-report.md").write_text(report)
(out / "extra-summary.json").write_text(json.dumps(summary, indent=2))
print(report)
