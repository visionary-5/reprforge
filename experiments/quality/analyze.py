#!/usr/bin/env python3
"""Analysis for pooled-gallery-upgrade-matrix-v1.

usage: analyze.py raw-output/output-12845056 [raw-output/output-602112] -> analysis-output/
Writes metrics.json, metrics.md and by_collection.md to the requested output directory.
"""
from __future__ import annotations

import json
from pathlib import Path

ROUTE_LABEL = {
    "target_full": "full re-encode (reference)",
    "stale": "stale index (no-op)",
    "procrustes_bridge": "Procrustes bridge",
    "affine_bridge": "affine ridge bridge",
    "hot_refresh@matched": "hot-refresh, matched GPU",
    "hot_refresh@0.25": "hot-refresh 25%",
    "hot_refresh@0.5": "hot-refresh 50%",
    "bf16": "exact cut (BF16)",
    "int8_token": "cut INT8/token",
    "pca512_int8": "cut PCA-512 INT8",
    "pca256_int8": "cut PCA-256 INT8",
    "pca128_int8": "cut PCA-128 INT8",
    "pooled_target_full": "full re-encode + token pooling x3",
    "pooled_pca256_int8": "cut PCA-256 + token pooling x3",
}
ROUTE_ORDER = list(ROUTE_LABEL)


DROP_QUERY_COLLECTIONS = {"shiftproject"}   # 900 of 1,000 rows carry no query; pages stay in the gallery as distractors


def load(root: Path) -> dict:
    """Aggregate valid queries against stable target rankings."""
    import statistics
    import numpy as np

    res = json.loads((root / "result.json").read_text())
    rk = json.loads((root / "rankings.json").read_text())
    pq = json.loads((root / "per_query.json").read_text())
    mask_file = root.parent / "query_mask.json"
    collection = rk["collection"]
    keep = [c not in DROP_QUERY_COLLECTIONS for c in collection]
    if mask_file.exists():
        vm = json.loads(mask_file.read_text())
        # per-collection validity in sorted-holdout order, laid out in the runner's query order
        pos = {c: 0 for c in vm}
        for i, c in enumerate(collection):
            if c in vm:
                keep[i] = keep[i] and bool(vm[c]["holdout_valid"][pos[c]]); pos[c] += 1
    idx = [i for i, k in enumerate(keep) if k]
    coll_kept = [collection[i] for i in idx]
    seed = 20260907
    for ti, (tname, routes) in enumerate(rk["top20"].items()):
        ref = routes["target_full"]
        rows_by_route = {}
        for route, top in routes.items():
            base_ref = routes["pooled_target_full"] if route == "pooled_pca256_int8" and "pooled_target_full" in routes else ref
            rows = []
            for q in idx:
                r = dict(pq[tname][route][q])
                r10, t10 = set(top[q][:10]), set(base_ref[q][:10]); r5, t5 = set(top[q][:5]), set(base_ref[q][:5])
                r["top10_overlap"] = len(r10 & t10) / 10; r["top5_overlap"] = len(r5 & t5) / 5
                r["top1_agreement"] = float(top[q][0] == base_ref[q][0])
                rows.append(r)
            rows_by_route[route] = rows
            keys = [k for k in rows[0] if k != "rank"]
            m = {k: statistics.fmean(r[k] for r in rows) for k in keys}
            m["top1_flip_rate"] = 1.0 - m["top1_agreement"]
            res["targets"][tname]["metrics"][route] = m
            res["targets"][tname]["metrics_by_collection"][route] = {
                c: {k: statistics.fmean(r[k] for r, cc in zip(rows, coll_kept) if cc == c) for k in keys}
                for c in dict.fromkeys(coll_kept)}
        route_names = list(routes)
        boot = {}
        for route in route_names:
            if route == "target_full":
                continue
            boot[route] = {}
            for fi, f in enumerate(("ndcg_at_5", "top10_overlap", "recall_at_1")):
                diff = np.asarray([a[f] - b[f] for a, b in zip(rows_by_route[route], rows_by_route["target_full"])])
                rng = np.random.default_rng(seed + 1000 * ti + 10 * route_names.index(route) + fi)
                draws = diff[rng.integers(0, len(diff), size=(20_000, len(diff)))].mean(axis=1)
                boot[route][f] = {"estimate": float(diff.mean()), "percentile_95": [float(v) for v in np.quantile(draws, (0.025, 0.975))], "draws": 20_000}
        res["targets"][tname]["paired_bootstrap_vs_target_full"] = boot
    res["gallery"]["queries"] = {c: coll_kept.count(c) for c in dict.fromkeys(coll_kept)}
    res["gallery"]["queries_total"] = len(idx)
    res["gallery"]["queries_dropped"] = {"shiftproject": collection.count("shiftproject"), "null_query_rows_elsewhere": sum(1 for i, c in enumerate(collection) if c not in DROP_QUERY_COLLECTIONS and not keep[i])}
    res["_recomputed_from_per_query_and_rankings"] = True
    return res


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def ci(b: dict) -> str:
    lo, hi = b["percentile_95"]
    return f"[{lo:+.3f},{hi:+.3f}]"


def matrix_md(res: dict) -> str:
    out = [f"# Pooled gallery: {res['gallery']['pages']} pages, {res['gallery']['queries_total']} queries, max_pixels {res['processor']['max_pixels']}", ""]
    out.append(f"Visual tokens/page {res['tokens_per_page']['visual']:.0f}; cut leverage " + ", ".join(f"{k} {v:.3f}" for k, v in res["cut_leverage"].items()))
    out.append("")
    for tname, t in res["targets"].items():
        out.append(f"## {tname} ({t['kind']}); matched hot-refresh fraction {t['suffix_share_matched_fraction']:.3f}")
        out.append("")
        out.append("| route | nDCG@5 | dnDCG@5 [95%] | R@1 | top-1 flip | TA@5 | TA@10 | payload B/page | GPU s/page |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in ROUTE_ORDER:
            if r not in t["metrics"]:
                continue
            m = t["metrics"][r]
            d = t["paired_bootstrap_vs_target_full"].get(r, {}).get("ndcg_at_5")
            dd = f"{d['estimate']:+.3f} {ci(d)}" if d else "—"
            payload = res["payload_bytes_per_page"].get(r)
            pb = f"{payload / 1e6:.2f} MB" if payload else ("—")
            gpu = res["stage_seconds_per_page"]
            if r == "target_full":
                sec = gpu["preprocess"] + gpu["prefix"] + gpu[f"suffix_target/{tname}"]
            elif r in res["payload_bytes_per_page"]:
                sec = res["replay_seconds_per_page"].get(f"{tname}/{r}", float("nan"))
            elif r.startswith("hot_refresh"):
                f = t["hot_refresh_fractions"][r.split("@")[1]]
                sec = f * (gpu["preprocess"] + gpu["prefix"] + gpu[f"suffix_target/{tname}"])
            else:
                sec = 0.0
            out.append(f"| {ROUTE_LABEL[r]} | {fmt(m['ndcg_at_5'])} | {dd} | {fmt(m['recall_at_1'])} | {fmt(m['top1_flip_rate'])} | {fmt(m['top5_overlap'])} | {fmt(m['top10_overlap'])} | {pb} | {sec:.3f} |")
        out.append("")
    tb = res["terminal_bytes_per_page"]
    out.append("## Storage denominators (bytes per page)")
    out.append("")
    out.append("| object | bytes/page |")
    out.append("|---|---|")
    for k, v in tb.items():
        out.append(f"| terminal {k} | {v / 1e6:.3f} MB |")
    for c, v in res["payload_bytes_per_page"].items():
        out.append(f"| cut {c} | {v / 1e6:.3f} MB |")
    out.append("")
    return "\n".join(out)


def by_collection_md(res: dict) -> str:
    out = ["# Per-collection nDCG@5 / top-1 flip / TA@10", ""]
    cols = list(res["gallery"]["queries"])
    for tname, t in res["targets"].items():
        out.append(f"## {tname}")
        out.append("")
        out.append("| route | " + " | ".join(f"{c} (n={res['gallery']['queries'][c]})" for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for r in ROUTE_ORDER:
            if r not in t["metrics_by_collection"]:
                continue
            cells = []
            for c in cols:
                m = t["metrics_by_collection"][r][c]
                cells.append(f"{m['ndcg_at_5']:.3f} / {1 - m['top1_agreement']:.2f} / {m['top10_overlap']:.2f}")
            out.append(f"| {ROUTE_LABEL[r]} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate retrieval quality from your run.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "metrics.md").write_text(matrix_md(result))
    (args.output / "by_collection.md").write_text(by_collection_md(result))


if __name__ == "__main__":
    main()
