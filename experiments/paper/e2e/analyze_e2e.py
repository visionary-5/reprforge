#!/usr/bin/env python3
"""Aggregate the end-to-end rebuild run: per-route wall-clock decomposition, cumulative scaling, and projections."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
targets = ["vidore-v0.2", "metric-ai-3b", "tsystems-3b", "colnomic-3b"]
def load(name):
    p = root / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
out = {"root": str(root)}
corpus = json.loads((root / "corpus.json").read_text())
out["corpus"] = {"pages": corpus["unique_pages"], "per_collection": corpus["per_collection"], "duplicates_dropped": corpus["duplicates_dropped"]}
rows = {}
for name in ["source-plain", "source-retain"] + [f"target-full-{t}" for t in targets] + [f"target-reprforge-{t}" for t in targets]:
    d = load(name)
    if not d:
        continue
    n = d["pages"]
    rows[name] = {"pages": n, "process_wall_s": d["process_wall_seconds"], "model_load_s": d["model_load_seconds"], "loop_s": d["loop_seconds"],
                  "seal_publish_s": d["seal_publish_seconds"], "validation": d.get("validation") or {}, "loop_ms_per_page": 1000 * d["loop_seconds"] / n,
                  "stage_ms_per_page": {k: 1000 * v / n for k, v in d["stage_totals_seconds"].items()}, "cumulative": d["cumulative_loop_seconds_at"],
                  "vector_GB": d["vector_bytes"] / 1e9, "state_GB": d.get("state_bytes_total", 0) / 1e9}
out["routes"] = rows
ver = {}
for t in targets:
    d = load(f"verify-{t}")
    if d:
        ver[t] = {k: d[k] for k in ("pages", "bitwise_equal_pages", "elements", "equal_elements", "max_abs_error", "mismatched_count")}
out["verification"] = ver
# lifecycle totals: one initial build + U upgrades, both routes, whole-process wall-clock
sr, sp = rows.get("source-retain"), rows.get("source-plain")
full = {t: rows.get(f"target-full-{t}") for t in targets}
rf = {t: rows.get(f"target-reprforge-{t}") for t in targets}
if sr and all(full.values()) and all(rf.values()):
    n = sr["pages"]
    plain_per_page = sp["loop_s"] / sp["pages"] if sp else None
    life = {"pages": n, "initial_build_plain_est_s": (plain_per_page * n + sp["model_load_s"] + sp["seal_publish_s"]) if sp else None,
            "initial_build_retain_s": sr["process_wall_s"], "upgrade_full_s": {t: full[t]["process_wall_s"] for t in targets},
            "upgrade_reprforge_s": {t: rf[t]["process_wall_s"] for t in targets}}
    life["speedup_per_upgrade"] = {t: full[t]["process_wall_s"] / rf[t]["process_wall_s"] for t in targets}
    life["four_upgrades_full_s"] = sum(life["upgrade_full_s"].values()); life["four_upgrades_reprforge_s"] = sum(life["upgrade_reprforge_s"].values())
    if life["initial_build_plain_est_s"]:
        life["lifecycle_plain_plus_full_s"] = life["initial_build_plain_est_s"] + life["four_upgrades_full_s"]
        life["lifecycle_retain_plus_reprforge_s"] = life["initial_build_retain_s"] + life["four_upgrades_reprforge_s"]
        life["lifecycle_reduction"] = 1 - life["lifecycle_retain_plus_reprforge_s"] / life["lifecycle_plain_plus_full_s"]
        life["retention_overhead_per_page_ms"] = 1000 * (sr["loop_s"] / n - plain_per_page)
        life["retention_overhead_fraction"] = (sr["loop_s"] / n) / plain_per_page - 1
    life["state_GB"] = sr["state_GB"]; life["state_MiB_per_page"] = sr["state_GB"] * 1e9 / n / 2**20
    out["lifecycle"] = life
    # linear scaling model fitted on cumulative checkpoints; projections to 1e5 and 1e6 pages
    proj = {}
    for label, r in [("full", full["vidore-v0.2"]), ("reprforge", rf["vidore-v0.2"]), ("retain", sr)]:
        cum = {int(k): v for k, v in r["cumulative"].items()}
        ks = sorted(cum)
        per_page = [cum[k] / k for k in ks]
        fixed = r["model_load_s"] + r["seal_publish_s"] + (r["validation"].get("seconds", 0) if r["validation"] else 0)
        slope = r["loop_s"] / r["pages"]
        proj[label] = {"checkpoints": {k: {"loop_s": cum[k], "s_per_page": cum[k] / k} for k in ks}, "per_page_spread": (max(per_page) - min(per_page)) / min(per_page),
                       "fixed_s": fixed, "slope_s_per_page": slope,
                       "projected_hours": {N: (fixed + slope * N) / 3600 for N in (1e4, 1e5, 1e6)}}
    proj["storage_TB_at"] = {N: life["state_MiB_per_page"] * N * 2**20 / 1e12 for N in (1e4, 1e5, 1e6)}
    out["scaling"] = proj
(root / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k in ("corpus", "verification", "lifecycle")}, indent=1, default=str))
for name, r in rows.items():
    print(f"{name:32s} pages={r['pages']:6d} wall={r['process_wall_s']:9.1f}s loop={r['loop_s']:9.1f}s ({r['loop_ms_per_page']:.1f} ms/page) load={r['model_load_s']:.1f}s seal={r['seal_publish_s']:.1f}s", {k: round(v, 1) for k, v in r['stage_ms_per_page'].items()})
if "scaling" in out:
    for label, p in out["scaling"].items():
        if label != "storage_TB_at":
            print(label, {k: round(v['s_per_page'], 4) for k, v in p['checkpoints'].items()}, "spread", round(p['per_page_spread'], 4), "hours@1e5/1e6", {k: round(v, 2) for k, v in p['projected_hours'].items()})
    print("storage TB", out["scaling"]["storage_TB_at"])
