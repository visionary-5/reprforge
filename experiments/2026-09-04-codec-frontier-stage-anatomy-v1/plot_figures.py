"""Paper figures from analysis-output/summary.json and extra-summary.json.

Figure A: fidelity--storage frontier (TA@10 vs retained bytes / fp32 terminal).
Figure B: cut leverage vs measured replay saving across corpora, backbones and
batch sizes; points on the diagonal mean the saving is exactly the leverage.
Outputs PDF into ../../paper/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / ".." / ".." / "paper" / "figures"
S = json.loads((ROOT / "analysis-output" / "summary.json").read_text())
E = json.loads((ROOT / "analysis-output" / "extra-summary.json").read_text())

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure A: frontier, two backbones
R25 = json.loads((ROOT / "raw-output" / "result.json").read_text())["results"]
R25["infovqa"] = json.loads((ROOT / "raw-output" / "output-infovqa" / "result.json").read_text())["results"]["infovqa"]
R2 = json.loads((ROOT / "raw-output" / "output-colqwen2" / "result.json").read_text())["results"]
codecs = ["bf16", "int8_token", "int4_group64", "pca512_int8", "pca256_int8", "pca128_int8", "pca64_int8"]
labels = {"bf16": "exact", "int8_token": "INT8", "int4_group64": "INT4", "pca512_int8": "PCA-512",
          "pca256_int8": "PCA-256", "pca128_int8": "PCA-128", "pca64_int8": "PCA-64"}
markers = {"arxivqa": "o", "docvqa": "s", "infovqa": "v", "flickr": "^"}
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax, R, title in ((axes[0], R25, "ColQwen2.5 v0.1→v0.2"), (axes[1], R2, "ColQwen2 v0.1→v1.0")):
    for corpus, m in markers.items():
        if corpus not in R:
            continue
        r = R[corpus]
        xs = [r["payload_bytes_per_page"][c] / r["terminal_bytes_per_page"]["float32"] for c in codecs]
        ys = [r["metrics"][c]["top10_overlap"] for c in codecs]
        ax.plot(xs, ys, marker=m, ms=3.5, lw=1, color="black", alpha=0.85, label=corpus)
        ax.axhline(r["metrics"]["stale"]["top10_overlap"], color="gray", lw=0.6, ls=":")
    r = R["arxivqa"]
    for c in codecs:
        x = r["payload_bytes_per_page"][c] / r["terminal_bytes_per_page"]["float32"]; y = r["metrics"][c]["top10_overlap"]
        ax.annotate(labels[c], (x, y), textcoords="offset points", xytext=(4, -9 if c != "pca64_int8" else 4), fontsize=6.5)
    ax.axhline(-1, color="gray", lw=0.6, ls=":", label="stale index")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("retained bytes / float32 terminal index")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("TA@10 vs exact target index")
axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "frontier.pdf")

# ---- Figure B: leverage vs saving
fig, ax = plt.subplots(figsize=(3.0, 2.4))
pts = []
for corpus, a in S["anatomy"].items():
    pts.append((a["leverage"], a["measured_saving_vs_raw_encode"], f"ColQwen2.5 {corpus} b1", "black", "o"))
for corpus, b in E["batch4"].items():
    pts.append((b["leverage"], b["saving"], f"ColQwen2.5 {corpus} b{b['batch']}", "black", "D"))
SB = json.loads((ROOT / "analysis-output" / "second-backbone-summary.json").read_text())
pts.append((SB["infovqa"]["leverage"], SB["infovqa"]["saving"], "ColQwen2.5 infovqa b1", "black", "o"))
for corpus, b in SB["colqwen2"].items():
    pts.append((b["leverage"], b["saving"], f"ColQwen2 {corpus}", "tab:blue", "x"))
OFFSETS = {"arxivqa b1": (6, -9), "docvqa b1": (-40, 6), "flickr b1": (6, 3), "infovqa b1": (-42, -10),
           "arxivqa b4": (6, -2), "docvqa b2": (-38, -14), "flickr b4": (6, -9),
           "ColQwen2 arxivqa": (-52, 4), "ColQwen2 docvqa": (6, 2), "ColQwen2 flickr": (6, -4), "ColQwen2 flickr b1": (6, -9)}
for x, y, name, color, marker in pts:
    ax.scatter(x, y, s=18, color=color, marker=marker, zorder=3)
    short = name.replace("ColQwen2.5 ", "")
    if "flickr" in short:  # only the small-image points need individual labels
        ax.annotate(short.replace("ColQwen2 flickr", "ColQwen2 flickr b1"), (x, y), textcoords="offset points",
                    xytext=OFFSETS.get(short, (5, 3)), fontsize=6)
ax.annotate("document pages,\nboth backbones,\nbatch 1/2/4", (0.885, 0.885), textcoords="offset points", xytext=(28, -34),
            fontsize=6, ha="center", arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "gray"})
ax.plot([0.35, 1.0], [0.35, 1.0], color="gray", lw=0.8, ls="--", zorder=1)
# 4090 Flickr from the paper's Table 6 (saving only; leverage not measured there)
ax.set_xlabel("cut leverage (pre-cut share of raw encode)")
ax.set_ylabel("measured replay saving")
ax.set_xlim(0.38, 0.95)
ax.set_ylim(0.38, 0.95)
ax.set_aspect("equal")
ax.text(0.60, 0.565, "saving = leverage", color="gray", fontsize=6.5, rotation=45, ha="left", va="bottom")
ax.scatter([], [], color="tab:blue", marker="x", label="ColQwen2"); ax.scatter([], [], color="black", marker="o", label="ColQwen2.5"); ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "leverage.pdf")
print("wrote", OUT / "frontier.pdf", OUT / "leverage.pdf")
