#!/usr/bin/env python3
"""Position-sensitive agreement and clustered uncertainty for a pooled-gallery run.

usage: extras.py raw-output/<run> -> analysis-output/extras-<max_pixels>.md and .json

Adds, per target and route:
  RBO@20 (p = 0.9)      rank-biased overlap of the route's top-20 with the target's top-20 (stable order)
  Kendall-tau@10        tau on the intersection of the two top-10 lists (positions), averaged over queries
  TA@10, TA@1           as before, for reference
  nDCG@5 difference vs target with a PAGE-CLUSTERED bootstrap (queries sharing a gold page are resampled
                        together; 20,000 draws) and the non-inferiority decision at the -0.01 floor made
                        on the LOWER 95% bound instead of the point estimate.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import DROP_QUERY_COLLECTIONS, load  # noqa: E402


def rbo(a: list[int], b: list[int], p: float = 0.9) -> float:
    """Rank-biased overlap (Webber et al. 2010), truncated at the shorter list, extrapolated."""
    k = min(len(a), len(b))
    if k == 0:
        return 0.0
    sa, sb = set(), set()
    overlap = 0.0; total = 0.0
    for d in range(1, k + 1):
        sa.add(a[d - 1]); sb.add(b[d - 1])
        x = len(sa & sb)
        total += (x / d) * p ** (d - 1)
    x_k = len(sa & sb)
    return (1 - p) * total + (x_k / k) * p ** k


def kendall_on_intersection(a: list[int], b: list[int]) -> float | None:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    pa = {x: i for i, x in enumerate(a)}; pb = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s = (pa[common[i]] - pa[common[j]]) * (pb[common[i]] - pb[common[j]])
            conc += s > 0; disc += s < 0
    return (conc - disc) / (conc + disc) if conc + disc else None


def main() -> None:
    root = Path(sys.argv[1])
    res = load(root)
    rk = json.loads((root / "rankings.json").read_text())
    pq = json.loads((root / "per_query.json").read_text())
    collection = rk["collection"]; gold = rk["gold"]
    keep = [c not in DROP_QUERY_COLLECTIONS for c in collection]
    mask_file = root.parent / "query_mask.json"
    if mask_file.exists():
        vm = json.loads(mask_file.read_text()); pos = {c: 0 for c in vm}
        for i, c in enumerate(collection):
            if c in vm:
                keep[i] = keep[i] and bool(vm[c]["holdout_valid"][pos[c]]); pos[c] += 1
    idx = [i for i, k in enumerate(keep) if k]
    clusters = defaultdict(list)
    for i in idx:
        clusters[gold[i]].append(i)
    cluster_list = list(clusters.values())
    rng = np.random.default_rng(20260909)
    draws = 20_000
    out = {"pages_with_queries": len(cluster_list), "queries": len(idx), "targets": {}}
    lines = [f"# Position-sensitive agreement and page-clustered intervals ({res['processor']['max_pixels']} px)", "",
             f"{len(idx)} queries on {len(cluster_list)} distinct gold pages; clustered bootstrap resamples pages.", ""]
    for tname, routes in rk["top20"].items():
        ref = routes["target_full"]
        tgt_rows = pq[tname]["target_full"]
        lines += [f"## {tname}", "", "| route | TA@1 | TA@10 | RBO@20 (p=.9) | Kendall τ@10 | ΔnDCG@5 | clustered 95% | NI at −.01 (lower bound) |", "|---|---|---|---|---|---|---|---|"]
        out["targets"][tname] = {}
        for route, top in routes.items():
            if route.startswith("pooled_"):
                continue
            ta1 = np.mean([top[q][0] == ref[q][0] for q in idx]); ta10 = np.mean([len(set(top[q][:10]) & set(ref[q][:10])) / 10 for q in idx])
            r = np.mean([rbo(top[q][:20], ref[q][:20]) for q in idx])
            ks = [kendall_on_intersection(top[q][:10], ref[q][:10]) for q in idx]; ks = [k for k in ks if k is not None]
            tau = float(np.mean(ks)) if ks else float("nan")
            rows = pq[tname][route]
            diff_q = np.asarray([rows[q]["ndcg_at_5"] - tgt_rows[q]["ndcg_at_5"] for q in range(len(rows))])
            # clustered bootstrap: resample gold pages with replacement, take all their queries
            cl = [np.asarray(c) for c in cluster_list]
            means = np.empty(draws)
            n_cl = len(cl)
            for d in range(draws):
                pick = rng.integers(0, n_cl, size=n_cl)
                sel = np.concatenate([cl[j] for j in pick])
                means[d] = diff_q[sel].mean()
            est = float(diff_q[idx].mean()); lo, hi = np.quantile(means, (0.025, 0.975))
            ni = "pass" if lo >= -0.01 else ("fail" if hi < -0.01 else "inconclusive")
            out["targets"][tname][route] = {"ta1": float(ta1), "ta10": float(ta10), "rbo20": float(r), "kendall10": tau, "dndcg5": est, "ci95_clustered": [float(lo), float(hi)], "noninferiority_lower_bound": ni}
            lines.append(f"| {route} | {ta1:.3f} | {ta10:.3f} | {r:.3f} | {tau:.3f} | {est:+.3f} | [{lo:+.3f},{hi:+.3f}] | {ni} |")
        lines.append("")
    outdir = HERE / "analysis-output"; outdir.mkdir(exist_ok=True)
    tag = res["processor"]["max_pixels"]
    (outdir / f"extras-{tag}.md").write_text("\n".join(lines) + "\n")
    (outdir / f"extras-{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
