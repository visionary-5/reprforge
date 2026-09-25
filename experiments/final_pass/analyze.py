"""Analyze recorded quality, timing and manual-interface comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    data = root / "quality-full"
    result = load(data / "result.json")
    independent = load(data / "independent-metrics.json")
    assert all(
        r["queries"] == 1908 and r["max_metric_error"] < 1e-12
        for r in independent.values()
    )
    assert not result["smoke_only"] and result["all_exact"]
    assert result["pages"] == 1110 and result["queries"] == 1908
    timing = [json.loads(s) for s in (data / "timing.jsonl").read_text().splitlines()]
    enc = [json.loads(s) for s in (data / "encoding.jsonl").read_text().splitlines()]
    assert len(timing) == 1800 and all(r["exact"] for r in timing)
    out = root / "analysis-output"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    routes = ["full", "generic", "fixed"]
    stages = [
        "decode_preprocess",
        "read_validate",
        "h2d_and_placeholders",
        "input_hash",
        "compute_excluding_hash",
        "write",
    ]
    stats = {}
    for route in routes:
        records = [r for r in timing if r["route"] == route]
        passes = [
            sum(r["total"] for r in records if r["repeat"] == rep) for rep in range(3)
        ]
        assert all(sum(r["repeat"] == rep for r in records) == 200 for rep in range(3))
        stats[route] = {
            "pass_seconds": passes,
            "ms_per_page_mean": float(np.mean(passes) * 5),
            "ms_per_page_sd": float(np.std(passes, ddof=1) * 5),
            "stages_ms": {
                s: float(np.mean([r[s] for r in records]) * 1000) for s in stages
            },
        }
    for route in routes[1:]:
        stats[route]["net_speedup"] = sum(stats["full"]["pass_seconds"]) / sum(
            stats[route]["pass_seconds"]
        )
        stats[route]["speedup_with_one_time_check"] = sum(
            stats["full"]["pass_seconds"]
        ) / (
            sum(stats[route]["pass_seconds"])
            + 3 * result["one_time_target_check_seconds"]
        )
    np_native = load(data / "native-per-query.json")
    np_common = load(data / "common-per-query.json")
    assert [r["id"] for r in np_native] == [r["id"] for r in np_common]
    deltas = np.array(
        [c["ndcg_at_10"] - n["ndcg_at_10"] for n, c in zip(np_native, np_common)]
    )
    stats["quality_delta_at_10"] = {
        "mean": float(deltas.mean()),
        "improved": int((deltas > 0).sum()),
        "unchanged": int((deltas == 0).sum()),
        "decreased": int((deltas < 0).sum()),
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2))
    quality_rows = []
    for route in ["native", "common", "replay"]:
        config = "native" if route == "native" else "common"
        tokens = np.array([r[config + "_visual_tokens"] for r in enc])
        secs = sum(r[route + "_seconds"] for r in enc)
        m = result["metrics"][route]
        quality_rows.append(
            f"| {route} | {m['ndcg_at_5']:.6f} | {m['ndcg_at_10']:.6f} | {tokens.mean():.1f} ({tokens.min()}–{tokens.max()}) | {secs:.2f} |"
        )
    quality_table = (
        "| Route | nDCG@5 | nDCG@10 | Actual visual tokens, mean (range) | Encoding seconds |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(quality_rows)
    )
    timing_table = (
        "| Route | ms/page, mean ± sample SD | Speedup | Speedup incl. one-time check |\n|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {r} | {stats[r]['ms_per_page_mean']:.3f} ± {stats[r]['ms_per_page_sd']:.3f} | {stats[r].get('net_speedup', 1):.3f}× | {stats[r].get('speedup_with_one_time_check', 1):.3f}× |"
            for r in routes
        )
    )
    manual_rows = []
    for architecture in ["qwen25", "qwen3"]:
        m = load(root / f"manual-{architecture}-full/result.json")
        assert m["interface_equal"] and m["dependencies_equal"]
        assert all(
            r["manual_exact"]
            and r["auto_exact"]
            and r["manual_max_error"] == 0
            and r["auto_max_error"] == 0
            for r in m["rows"]
        )
        manual_rows.append(
            f"| {architecture} | {len(m['dependencies'])} | {len(m['rows'])}/{len(m['rows'])} | {len(m['rows'])}/{len(m['rows'])} | {m['trace_seconds']:.3f} |"
        )
    manual_table = (
        "| Architecture | Dependencies | Manual exact | Automatic exact | Trace seconds |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(manual_rows)
    )
    report = (
        "# Final-pass evidence\n\n"
        + quality_table
        + "\n\n"
        + timing_table
        + "\n\n"
        + manual_table
        + f"""\n
Target: `nomic-ai/colnomic-embed-multimodal-3b`, revision
`86627b4a9b0cade577851a70afa469084f9863a4`. Qwen base revision:
`66285546d2b821cf421d4f5eb2576359d3770cd3`; exact weight hashes are in
the public-input specifications and recorded model revisions supplied with this supplement. Native uses the released
1,003,520-pixel processor; common uses the manuscript base processor at 602,112.

Quality uses the full ViDoRe v3 HR test corpus: 1,110 pages and all 1,908
standard queries. Independently computed pytrec_eval metrics agree to within
3.34e-16 per query. Native and common query banks were independently encoded
with their respective processors and are bitwise equal on all 1,908 queries.
Common replay
matches every page and the complete common score matrix exactly. The observed
common-minus-native nDCG@10 difference is {deltas.mean():+.6f}; this is a measured
tradeoff on one split, not evidence of unchanged native quality on every task.
Encoding totals include preprocessing and vector writes; source retention,
model loading, dependency hashing and retrieval scoring are separate costs.
These encoding totals are one pass, in native/common/replay order per page; the
separate three-pass paired timing supplies the validation-cost comparison.

Timing is three paired passes over 200 HR pages, with warm retained-state reads,
resident compressed image bytes, BF16, batch one, the same target forward and
vector writes/fsync. Each mean/SD uses three pass means, not 600 independent pages.
Stage intervals include synchronization and instrumentation, so even skipped
stages can have small nonzero timer overhead. The shared GPU had a resident
VLLM process; the saved process monitor recorded no nonzero SM-utilization
samples for that process during this run. This is not an isolated-host benchmark.
One-time target dependency checking took {result["one_time_target_check_seconds"]:.6f} s.
This sample must not replace the historical 20,946-page lifecycle numbers.

Manual Qwen2.5 tests source v0.1 to v0.2; Qwen3 tests the Tomoro/OpenSearch 4B
wrappers. The table reports source-trace time only; target tracing was performed
but not separately timed in these runs. Trace seconds cover the complete traced
forward after one warmup, not the incremental tracing overhead over a timed plain
forward. Both use BF16 and batch one.

Manual and automatic routes share capture, serialization, dependency checking and
resume. Expert rules name the visual module, include all named buffers, select
the appropriate output leaves, and map the Qwen3 wrapper prefixes. Automatic
discovery extracts the interface and dependencies from the observed execution;
model loading, seed selection, operator semantics and execution assumptions
remain integration responsibilities. No replay-speed advantage over the correct
manual baseline is claimed.

| Integration | Complete manual rules | Automatic route still supplied by integration |
|---|---|---|
| Qwen2.5 | `visual` output leaf 0; enumerate every parameter and registered buffer under `visual`, including non-persistent RoPE buffer | Pixel/text/grid/mask seeds; publisher loader and retrieval projection; fixed operator/configuration assumptions |
| Qwen3 4B | `vlm.model.visual` leaves 1–4, including all three DeepStack outputs; all visual parameters/buffers | Pixel/text/structural seed mapping; publisher loader adaptations; explicit source/target prefix mapping shared by both routes |

The comparison checks agreement of the tested interface and dependency sets.
It does not measure integration effort or engineering time.

"""
    )
    (out / "analysis-report.md").write_text(report)
    (out / "stats-appendix.md").write_text("""# Statistical scope

The quality result is a deterministic, complete evaluation of this frozen split.
Queries sharing evidence and translated variants are not independent population
samples. We report paired score differences and their exact empirical counts;
no population confidence interval, significance test or equivalence claim is
justified by this experiment. No multiple-comparison p-values are produced.

Timing uses three repeated passes on the same 200 pages and shared A100 host.
Sample SD describes pass-to-pass variation; it is not a confidence interval over
hardware or workloads. Net speedup is the ratio of summed paired times. The
one-time-check variant adds one target dependency check to each 200-page pass.
Model load and source-retention costs remain excluded from that sample ratio.

Exactness is an exhaustive deterministic check on the measured page set, not
a statistical claim about unobserved branches or numerical environments.
""")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    vals = [result["metrics"][r]["ndcg_at_10"] for r in ["native", "common", "replay"]]
    axes[0].bar(
        ["Native", "Common full", "Common replay"],
        vals,
        color=["#536878", "#327da8", "#65a7ba"],
    )
    axes[0].set(
        ylabel="nDCG@10",
        ylim=(0, 1),
        title="Complete HR split: 1,110 pages / 1,908 queries",
    )
    axes[1].hist(deltas, bins=35, color="#327da8")
    axes[1].set(
        xlabel="Common minus native nDCG@10 per query",
        ylabel="Queries",
        title="Observed paired differences",
    )
    for ext in ["png", "svg"]:
        fig.savefig(out / "figures" / f"quality.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    bottom = np.zeros(3)
    for stage in stages:
        values = np.array([stats[r]["stages_ms"][stage] for r in routes])
        ax.bar(routes, values, bottom=bottom, label=stage.replace("_", " "))
        bottom += values
    ax.set(
        ylabel="Milliseconds per page",
        title="200 pages × 3 paired passes; warm state reads",
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    for ext in ["png", "svg"]:
        fig.savefig(out / "figures" / f"timing.{ext}")
    plt.close(fig)
    (out / "figure-catalog.md").write_text("""# Figures

- `quality.png/svg`: complete-split quality and paired query differences. No
  uncertainty bars or significance claims. Purpose: distinguish exact recovery
  of common target representations from preservation of native target quality.
- `timing.png/svg`: measured stage means of three 200-page paired passes.
  Purpose: show validation cost and distinguish generic and fixed-contract net
  benefits. Warm reads, shared host, model checks excluded; not lifecycle timing.
""")


if __name__ == "__main__":
    main()
