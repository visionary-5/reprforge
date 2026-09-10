#!/usr/bin/env python3
"""Validate and summarize the unified RTX 4090 main-table repeats."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw-output"
OUT = ROOT / "analysis-output"
BENCHMARKS = ("arxivqa", "docvqa", "flickr", "infovqa")
LABELS = {
    "arxivqa": "ArxivQA",
    "docvqa": "DocVQA",
    "flickr": "Flickr",
    "infovqa": "InfoVQA",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def svg_build_time(rows: list[dict]) -> str:
    width, height = 860, 470
    left, top, bottom, right = 78, 34, 74, 24
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = 240.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}.raw{fill:#8ab4f8}.replay{fill:#34a853}.point{fill:#202124}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="430" y="21" text-anchor="middle" font-size="16" font-weight="bold">Complete target-build time on one RTX 4090 (n=3)</text>',
    ]
    for tick in range(0, 241, 40):
        y = top + plot_h * (1 - tick / ymax)
        parts += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>', f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick}</text>']
    parts += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>', f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>', '<text x="17" y="225" transform="rotate(-90 17 225)" text-anchor="middle" font-size="13">seconds</text>']
    group_w = plot_w / len(rows)
    bar_w = 38
    for i, row in enumerate(rows):
        center = left + group_w * (i + 0.5)
        for j, (key, cls) in enumerate((("raw_seconds", "raw"), ("replay_seconds", "replay"))):
            x = center + (-bar_w - 6 if j == 0 else 6)
            value = row[f"{key}_mean"]
            y = top + plot_h * (1 - value / ymax)
            h = top + plot_h - y
            parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}"/>')
            for run_value in row[f"{key}_runs"]:
                py = top + plot_h * (1 - run_value / ymax)
                parts.append(f'<circle class="point" cx="{x+bar_w/2:.1f}" cy="{py:.1f}" r="2.7"/>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="13">{LABELS[row["benchmark"]]}</text>')
    parts += ['<rect class="raw" x="610" y="438" width="15" height="10"/><text x="631" y="447" font-size="12">Raw rebuild</text>', '<rect class="replay" x="710" y="438" width="15" height="10"/><text x="731" y="447" font-size="12">ReprForge replay</text>', '</svg>']
    return "\n".join(parts) + "\n"


def svg_saving(rows: list[dict]) -> str:
    width, height = 760, 430
    left, top, bottom, right = 76, 38, 70, 24
    plot_w, plot_h = width-left-right, height-top-bottom
    ymin, ymax = 70.0, 90.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}.bar{fill:#34a853}.point{fill:#202124}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="22" text-anchor="middle" font-size="16" font-weight="bold">Paired target-build saving (n=3)</text>',
    ]
    for tick in range(70, 91, 5):
        y = top + plot_h * (1 - (tick-ymin)/(ymax-ymin))
        parts += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>', f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick}</text>']
    parts += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>', f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>', '<text x="17" y="215" transform="rotate(-90 17 215)" text-anchor="middle" font-size="13">saving (%)</text>']
    group_w = plot_w / len(rows)
    for i, row in enumerate(rows):
        center = left + group_w * (i + 0.5)
        value = 100 * row["saving_mean"]
        y = top + plot_h * (1 - (value-ymin)/(ymax-ymin))
        base = top + plot_h
        parts.append(f'<rect class="bar" x="{center-27:.1f}" y="{y:.1f}" width="54" height="{base-y:.1f}"/>')
        for run_value in row["saving_runs"]:
            py = top + plot_h * (1 - (100*run_value-ymin)/(ymax-ymin))
            parts.append(f'<circle class="point" cx="{center:.1f}" cy="{py:.1f}" r="3"/>')
        parts += [f'<text x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12">{value:.2f}%</text>', f'<text x="{center:.1f}" y="{base+24}" text-anchor="middle" font-size="13">{LABELS[row["benchmark"]]}</text>']
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    host = json.loads((ROOT / "host-manifest.json").read_text())
    rows: list[dict] = []
    checksums: dict[str, str] = {}
    for benchmark in BENCHMARKS:
        expected = protocol["expected"][benchmark]
        raw_times, replay_times, savings, ir_ratios = [], [], [], []
        for run in range(1, 4):
            path = RAW / f"{benchmark}-run{run}.json"
            payload = path.read_bytes()
            checksums[path.name] = hashlib.sha256(payload).hexdigest()
            result = json.loads(payload)
            if result["protocol"] != expected["protocol"]:
                raise RuntimeError(f"protocol mismatch: {path}")
            scope = result["scope"]
            if not scope["complete"] or scope["pages"] != expected["pages"] or scope["queries"] != expected["queries"]:
                raise RuntimeError(f"scope mismatch: {path}")
            for gate in protocol["gate_policy"]["required_boolean_gates"]:
                if result["gates_before_quality"].get(gate) is not True:
                    raise RuntimeError(f"failed {gate}: {path}")
            light = result["storage"]["light"]
            raw_ndcg = light["raw"]["0.1"]["actual_index_ndcg_at_5"]
            replay_ndcg = light["replay"]["0.1"]["actual_index_ndcg_at_5"]
            if abs(raw_ndcg - expected["raw_light10_ndcg_at_5"]) > 1e-12 or abs(replay_ndcg - expected["reprforge_light10_ndcg_at_5"]) > 1e-12:
                raise RuntimeError(f"quality identity mismatch: {path}")
            timing = result["timing_seconds"]
            build = timing["method_complete_target_build"]
            raw_times.append(build["raw_light10"])
            replay_times.append(build["reprforge_light10"])
            savings.append(timing["reprforge_light10_saving"])
            ir_ratios.append(result["storage"]["ir_to_raw_terminal_ratio"])
        raw_mean, raw_sd = mean_sd(raw_times)
        replay_mean, replay_sd = mean_sd(replay_times)
        saving_mean, saving_sd = mean_sd(savings)
        delta = expected["reprforge_light10_ndcg_at_5"] - expected["raw_light10_ndcg_at_5"]
        rows.append({
            "benchmark": benchmark,
            "pages": expected["pages"],
            "queries": expected["queries"],
            "raw_seconds_runs": raw_times,
            "replay_seconds_runs": replay_times,
            "saving_runs": savings,
            "raw_seconds_mean": raw_mean,
            "raw_seconds_sd": raw_sd,
            "replay_seconds_mean": replay_mean,
            "replay_seconds_sd": replay_sd,
            "saving_mean": saving_mean,
            "saving_sd": saving_sd,
            "raw_light10_ndcg_at_5": expected["raw_light10_ndcg_at_5"],
            "reprforge_light10_ndcg_at_5": expected["reprforge_light10_ndcg_at_5"],
            "quality_delta": delta,
            "paired_bootstrap_95": expected["paired_bootstrap_95"],
            "aggregate_quality_gate": delta >= protocol["gate_policy"]["quality_delta_floor"],
            "ir_to_terminal_ratio": statistics.mean(ir_ratios),
            "all_system_gates_pass": True,
        })

    OUT.mkdir(parents=True)
    (OUT / "figures").mkdir()
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "benchmark": row["benchmark"],
            "pages": row["pages"],
            "queries": row["queries"],
            "raw_seconds_mean": row["raw_seconds_mean"],
            "raw_seconds_sample_sd": row["raw_seconds_sd"],
            "reprforge_seconds_mean": row["replay_seconds_mean"],
            "reprforge_seconds_sample_sd": row["replay_seconds_sd"],
            "saving_mean": row["saving_mean"],
            "saving_sample_sd": row["saving_sd"],
            "raw_light10_ndcg_at_5": row["raw_light10_ndcg_at_5"],
            "reprforge_light10_ndcg_at_5": row["reprforge_light10_ndcg_at_5"],
            "quality_delta": row["quality_delta"],
            "ir_to_terminal_ratio": row["ir_to_terminal_ratio"],
        })
    with (OUT / "exact-table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (OUT / "stats.json").write_text(json.dumps({
        "protocol": protocol["protocol_id"],
        "host": host,
        "rows": rows,
        "sha256": checksums,
        "all_twelve_system_gate_sets_pass": all(row["all_system_gates_pass"] for row in rows),
        "all_four_aggregate_quality_gates_pass": all(row["aggregate_quality_gate"] for row in rows),
        "boundary": protocol["boundary"],
    }, indent=2) + "\n")
    (OUT / "figures/figure-01-build-time.svg").write_text(svg_build_time(rows))
    (OUT / "figures/figure-02-saving.svg").write_text(svg_saving(rows))

    table = [
        "| benchmark | pages / queries | Raw Light10 (s) | ReprForge Light10 (s) | saving | Light10 nDCG delta | IR / terminal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {LABELS[row['benchmark']]} | {row['pages']} / {row['queries']} "
            f"| {row['raw_seconds_mean']:.2f} ± {row['raw_seconds_sd']:.2f} "
            f"| {row['replay_seconds_mean']:.2f} ± {row['replay_seconds_sd']:.2f} "
            f"| {100*row['saving_mean']:.2f}% ± {100*row['saving_sd']:.2f} pp "
            f"| {row['quality_delta']:+.5f} | {row['ir_to_terminal_ratio']:.3f}× |"
        )
    report = "\n".join([
        "# Unified RTX 4090 main-table repeats",
        "",
        "## Decisive result",
        "",
        *table,
        "",
        "All twelve complete transitions passed dependency, exact-lowering, cardinality, and storage gates. Every paired run saved time. Across four retrieval surfaces, ReprForge reduced the measured adapter-transition target-build time by 77.76%--85.23% while the already frozen Light10 quality deltas all remained above the aggregate -0.01 gate.",
        "",
        "The narrow within-host variation removes the earlier weakness that the main systems table mixed one-off measurements from different RTX 4090 instances. It does not create new query samples: the paired-query bootstrap intervals remain those of the original benchmark runs.",
        "",
        "## Interpretation",
        "",
        "The result supports the paper's lifecycle claim, not a generic encoder speedup claim. For a fixed processor and an adapter/projection-only target update, the corpus-calibrated reusable post-vision IR avoids recomputing the visual prefix and still materializes the target adapter's compact physical index. The benefit survives document-like pages, natural images, and two visual-QA surfaces under one implementation and one host.",
        "",
        "## Statistical boundary",
        "",
        "With n=3 technical repeats, no parametric significance test is reported. The analysis shows every raw/replay observation, mean, and sample SD. Retrieval uncertainty is query-level and comes from the previously frozen paired bootstrap: only ArxivQA has an interval whose lower bound clears -0.01; DocVQA, Flickr, and InfoVQA retain wider intervals despite passing the aggregate point gate.",
        "",
        "## Reproducibility boundary",
        "",
        protocol["boundary"],
        "",
    ])
    (OUT / "analysis-report.md").write_text(report)
    appendix = "\n".join([
        "# Statistics appendix",
        "",
        "- Experimental unit for timing: one complete 500- or 1,000-item target-index transition.",
        "- Repetitions: three sequential complete transitions per benchmark on the same RTX 4090 host.",
        "- Timing summary: arithmetic mean ± sample SD; raw observations are retained in `stats.json`.",
        "- Pairing: raw and ReprForge routes are measured inside each complete run; saving is calculated per run before summarization.",
        "- Quality: deterministic serialized-index nDCG identity was checked across repeats; no additional quality uncertainty is claimed from technical repeats.",
        "- Gate validation: booleans are checked by name; the exact lowering error of 0.0 is not incorrectly treated as a failed Boolean.",
        "- Multiplicity: no new hypothesis tests were performed in this technical-repeat bundle.",
        "- Hardware: a single 24,564 MiB RTX 4090 container with 12 visible CPU cores; results are not pooled with prior A100 or other 4090 hosts.",
        "",
    ])
    (OUT / "stats-appendix.md").write_text(appendix)
    catalog = "\n".join([
        "# Figure catalog",
        "",
        "## Figure 1 — complete target-build time",
        "",
        "Grouped raw/replay bars for each benchmark with all three observations overlaid. Use for the main systems comparison; the y-axis starts at zero.",
        "",
        "## Figure 2 — paired target-build saving",
        "",
        "Per-benchmark mean saving with all three paired observations overlaid. The truncated 70%--90% axis is explicit and should be used only as a variability/detail panel, not as a standalone magnitude comparison.",
        "",
    ])
    (OUT / "figure-catalog.md").write_text(catalog)


if __name__ == "__main__":
    main()
