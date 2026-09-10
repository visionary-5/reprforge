#!/usr/bin/env python3
"""Analysis for colqwen2-pooled-upgrade-v1: matrix, exactness record, frozen predictions C1-C5.

usage: analyze.py raw-output/output
Reuses the pooled-gallery loader (dedup mask, stable reference, Shift drop) from the 2026-09-07 experiment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-07-pooled-gallery-upgrade-matrix-v1"))
from analyze import load, matrix_md, by_collection_md  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1])
    # the loader looks for query_mask.json next to the run directory
    mask_src = HERE.parent / "2026-09-07-pooled-gallery-upgrade-matrix-v1/raw-output/query_mask.json"
    mask_dst = root.parent / "query_mask.json"
    if not mask_dst.exists():
        mask_dst.write_text(mask_src.read_text())
    res = load(root)
    out = HERE / "analysis-output"; out.mkdir(exist_ok=True)
    (out / "matrix.md").write_text(matrix_md(res)); (out / "by_collection.md").write_text(by_collection_md(res))
    t = res["targets"]["colqwen2-v1.0"]; m = t["metrics"]; ex = res["exact_cut_terminal_exactness"]["colqwen2-v1.0"]
    preds = []
    preds.append(("C1_exact", abs(m["bf16"]["top10_overlap"] - 1) < 1e-9 and ex["bitwise_equal_pages"] >= 38 and ex["max_abs_error"] <= 1e-2,
                  f"exact TA@10 {m['bf16']['top10_overlap']:.3f}; bitwise-identical pages {ex['bitwise_equal_pages']}/{ex['pages']}; max abs error {ex['max_abs_error']:.2e}"))
    preds.append(("C2_stale_breaks", m["stale"]["top10_overlap"] <= 0.60 and m["stale"]["top1_flip_rate"] >= 0.25,
                  f"stale TA@10 {m['stale']['top10_overlap']:.3f}, top-1 flip {m['stale']['top1_flip_rate']:.3f}, dnDCG {m['stale']['ndcg_at_5'] - m['target_full']['ndcg_at_5']:+.3f}"))
    d512 = m["pca512_int8"]["ndcg_at_5"] - m["target_full"]["ndcg_at_5"]; d256 = m["pca256_int8"]["ndcg_at_5"] - m["target_full"]["ndcg_at_5"]
    preds.append(("C3_operating_point", m["pca512_int8"]["top10_overlap"] >= 0.85 and abs(d512) <= 0.01 and (d256 < -0.01 or m["pca256_int8"]["top10_overlap"] < 0.80),
                  f"PCA-512 TA {m['pca512_int8']['top10_overlap']:.3f} dnDCG {d512:+.3f}; PCA-256 TA {m['pca256_int8']['top10_overlap']:.3f} dnDCG {d256:+.3f}"))
    lev = res["cut_leverage"]["colqwen2-v1.0"]
    preds.append(("C4_leverage", 0.78 <= lev <= 0.92, f"leverage {lev:.3f}; tokens/page {res['tokens_per_page']['visual']:.0f}"))
    preds.append(("C5_hot_refresh", m["hot_refresh@matched"]["top10_overlap"] < 0.5, f"matched hot-refresh TA@10 {m['hot_refresh@matched']['top10_overlap']:.3f} at fraction {t['suffix_share_matched_fraction']:.3f}"))
    lines = ["# Frozen predictions (colqwen2-pooled-upgrade-v1)", "", "| id | holds | evidence |", "|---|---|---|"] + [f"| {i} | {h} | {e} |" for i, h, e in preds]
    (out / "predictions.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"exactness": ex, "leverage": lev, "predictions": [{"id": i, "holds": h, "evidence": e} for i, h, e in preds],
                                                   "metrics": m, "stage_seconds_per_page": res["stage_seconds_per_page"], "gallery": res["gallery"]}, indent=2) + "\n")
    print(matrix_md(res)); print("\n".join(lines))


if __name__ == "__main__":
    main()
