#!/usr/bin/env python3
"""Figure: four upgrades on the pooled gallery.
(a) nDCG@5 of full re-encode / stale / PCA-256 cut per upgrade. (b) TA@10 of every route against its GPU share.
Legends sit outside the axes so no label covers data."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load

res = load(Path(sys.argv[1])); out = Path(sys.argv[2])
T = res["targets"]; names = ["vidore-v0.2", "metric-ai-3b", "tsystems-3b", "colnomic-3b"]
labels = {"vidore-v0.2": "vidore\nv0.2", "metric-ai-3b": "Metric-\nAI", "tsystems-3b": "T-\nSystems", "colnomic-3b": "Col-\nNomic"}
legend_names = {"vidore-v0.2": "vidore v0.2", "metric-ai-3b": "Metric-AI", "tsystems-3b": "T-Systems", "colnomic-3b": "ColNomic"}
C = {"full": "#333333", "stale": "#c0392b", "bridge": "#e67e22", "hot": "#8e9aa0", "pca": "#2e86c1", "exact": "#1b4f72"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.32})

# (a) bars
x = list(range(len(names))); w = 0.27
series = [("target_full", "full re-encode", C["full"]), ("stale", "stale index", C["stale"]), ("pca256_int8", "PCA-256 cut", C["pca"])]
for i, (route, lab, col) in enumerate(series):
    vals = [T[n]["metrics"][route]["ndcg_at_5"] for n in names]
    bars = ax1.bar([v + (i - 1) * w for v in x], vals, w, label=lab, color=col)
    for b, v in zip(bars, vals):
        if v < 0.05:
            ax1.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=col, rotation=90)
ax1.set_xticks(x); ax1.set_xticklabels([labels[n] for n in names], fontsize=7.5)
ax1.set_ylabel("nDCG@5 on the pooled gallery"); ax1.set_ylim(0, 1.0)
ax1.set_title("(a) quality of the served index after the upgrade", fontsize=8, loc="left")

# (b) TA vs GPU share
markers = {"vidore-v0.2": "o", "metric-ai-3b": "s", "tsystems-3b": "^", "colnomic-3b": "D"}
for n in names:
    t = T[n]; g = res["stage_seconds_per_page"]; full = g["preprocess"] + g["prefix"] + g[f"suffix_target/{n}"]
    pts = [(0.0, t["metrics"]["stale"]["top10_overlap"], "stale"),
           (0.0, max(t["metrics"]["procrustes_bridge"]["top10_overlap"], t["metrics"]["affine_bridge"]["top10_overlap"]), "bridge")]
    for k, f in t["hot_refresh_fractions"].items():
        pts.append((f, t["metrics"][f"hot_refresh@{k}"]["top10_overlap"], "hot"))
    pts.append((res["replay_seconds_per_page"][f"{n}/pca256_int8"] / full, t["metrics"]["pca256_int8"]["top10_overlap"], "pca"))
    if "bf16" in t["metrics"]:
        pts.append((res["replay_seconds_per_page"][f"{n}/bf16"] / full, t["metrics"]["bf16"]["top10_overlap"], "exact"))
    pts.append((1.0, 1.0, "full"))
    for gx, ta, kind in pts:
        ax2.scatter(gx, ta, marker=markers[n], s=24, color=C[kind], edgecolor="white", linewidth=0.4, zorder=3)
ax2.axvline(0.1, color="#bbbbbb", lw=0.6, ls=":", zorder=1)
ax2.text(0.115, 0.03, "cut budget", fontsize=6.5, color="#777777")
ax2.set_xlabel("GPU time of the route (share of a full re-encode)"); ax2.set_ylabel("TA@10 against the target index")
ax2.set_ylim(-0.02, 1.06); ax2.set_xlim(-0.04, 1.06)
ax2.set_title("(b) fidelity to the target ranking against GPU cost", fontsize=8, loc="left")
route_handles = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=k) for k, c in
                 [("stale index", C["stale"]), ("best bridge", C["bridge"]), ("partial re-encode", C["hot"]), ("PCA-256 cut", C["pca"]), ("exact cut", C["exact"]), ("full re-encode", C["full"])]]
shape_handles = [Line2D([], [], marker=markers[n], ls="", color="#555555", markersize=5, label=legend_names[n]) for n in names]
bar_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in [("full re-encode", C["full"]), ("stale index", C["stale"]), ("PCA-256 cut", C["pca"])]]
fig.legend(handles=bar_handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.24, 0.02), ncol=3, handlelength=1.2, columnspacing=1.0, title="(a) bars", title_fontsize=7)
fig.legend(handles=route_handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.70, 0.02), ncol=3, handlelength=1.0, columnspacing=0.8, title="(b) route, by colour", title_fontsize=7)
fig.legend(handles=shape_handles, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.70, -0.13), ncol=4, handlelength=1.0, columnspacing=0.8, title="(b) upgrade, by marker", title_fontsize=7)
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
