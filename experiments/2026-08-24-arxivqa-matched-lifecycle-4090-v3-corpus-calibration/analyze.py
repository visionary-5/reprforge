#!/usr/bin/env python3
"""Strict analysis for the matched ArxivQA target-version lifecycle experiment."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


METHODS = (
    ("raw_full", "Raw Full", "raw", "full", 1.0),
    ("raw_light10", "Raw + Light 10%", "raw", "merge_tome", 0.10),
    ("raw_light05", "Raw + Light 5%", "raw", "merge_tome", 0.05),
    ("reprforge_full", "ReprForge Full", "replay", "full", 1.0),
    ("reprforge_light10", "ReprForge + Light 10%", "replay", "merge_tome", 0.10),
    ("reprforge_light05", "ReprForge + Light 5%", "replay", "merge_tome", 0.05),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--build-result", type=Path, required=True)
    parser.add_argument("--raw-baseline", type=Path, required=True)
    parser.add_argument("--replay-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def quality(baseline: dict, method: str, ratio: float) -> float:
    if method == "full":
        return float(baseline["count"]["full"]["1.0"]["ndcg"])
    return float(baseline["count"][method][str(ratio)]["ndcg"])


def observed_quality(build: dict, baseline: dict, source: str, method: str, ratio: float) -> float:
    if method == "full":
        return quality(baseline, method, ratio)
    return float(build["storage"]["light"][source][str(ratio)]["actual_index_ndcg_at_5"])


def target_bytes(build: dict, source: str, ratio: float) -> int:
    if ratio == 1.0:
        return int(build["storage"][f"{source}_full_terminal_bytes"])
    return int(build["storage"]["light"][source][str(ratio)]["bytes"])


def lifecycle_svg(path: Path, rows: list[dict]) -> None:
    width, height = 900, 440
    left, right, top, bottom = 100, 40, 55, 90
    plot_h = height - top - bottom
    groups = (("full", "Full", 1.0), ("light10", "Light 10%", 0.10), ("light05", "Light 5%", 0.05))
    raw = {row["method"]: row for row in rows if row["source"] == "raw"}
    replay = {row["method"]: row for row in rows if row["source"] == "replay"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_h * (1 - value)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>', f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>']
    for index, (suffix, label, _) in enumerate(groups):
        center = 220 + index * 245
        raw_row = raw[f"raw_{suffix}"]
        replay_row = replay[f"reprforge_{suffix}"]
        for offset, row, color in ((-30, raw_row, "#999999"), (30, replay_row, "#0072B2")):
            value = row["build_seconds"] / raw_row["build_seconds"]
            bar_h = value * plot_h
            parts.append(f'<rect x="{center+offset-22}" y="{top+plot_h-bar_h:.1f}" width="44" height="{bar_h:.1f}" fill="{color}" stroke="#111827"/>')
            parts.append(f'<text x="{center+offset}" y="{top+plot_h-bar_h-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}×</text>')
        parts.append(f'<text x="{center}" y="{height-bottom+30}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(label)}</text>')
    parts += [
        '<text transform="translate(24,230) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Target-build time / matched raw route</text>',
        '<rect x="585" y="18" width="16" height="16" fill="#999999" stroke="#111827"/><text x="608" y="31" font-family="sans-serif" font-size="12">Raw target build</text>',
        '<rect x="730" y="18" width="16" height="16" fill="#0072B2" stroke="#111827"/><text x="753" y="31" font-family="sans-serif" font-size="12">ReprForge replay</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts) + "\n")


def quality_svg(path: Path, rows: list[dict]) -> None:
    width, height = 900, 400
    left, right, top, bottom = 140, 50, 55, 65
    contrasts = [row for row in rows if row["source"] == "replay"]
    lower = min(-0.015, min(row["quality_delta_vs_raw"] for row in contrasts) - 0.003)
    upper = max(0.015, max(row["quality_delta_vs_raw"] for row in contrasts) + 0.003)

    def x(value: float) -> float:
        return left + (value - lower) / (upper - lower) * (width - left - right)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for value, color, dash in ((0.0, "#111827", ""), (-0.01, "#D55E00", ' stroke-dasharray="6,4"')):
        parts.append(f'<line x1="{x(value):.1f}" y1="{top}" x2="{x(value):.1f}" y2="{height-bottom}" stroke="{color}" stroke-width="2"{dash}/>')
    labels = {"reprforge_full": "Full", "reprforge_light10": "Light 10%", "reprforge_light05": "Light 5%"}
    for index, row in enumerate(contrasts):
        y = top + 55 + index * 75
        value = row["quality_delta_vs_raw"]
        parts.append(f'<text x="{left-15}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="15">{labels[row["method"]]}</text>')
        parts.append(f'<circle cx="{x(value):.1f}" cy="{y}" r="7" fill="#0072B2" stroke="#111827"/>')
        anchor = "start" if value >= 0 else "end"
        offset = 12 if value >= 0 else -12
        parts.append(f'<text x="{x(value)+offset:.1f}" y="{y+5}" text-anchor="{anchor}" font-family="sans-serif" font-size="13">{value:+.4f}</text>')
    for index in range(7):
        value = lower + index * (upper - lower) / 6
        parts.append(f'<line x1="{x(value):.1f}" y1="{height-bottom}" x2="{x(value):.1f}" y2="{height-bottom+6}" stroke="#111827"/>')
        parts.append(f'<text x="{x(value):.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:+.3f}</text>')
    parts += [
        f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="14">nDCG@5 difference: ReprForge route minus matched raw route</text>',
        f'<text x="{x(-0.01)+6:.1f}" y="{top+15}" font-family="sans-serif" font-size="11" fill="#D55E00">primary non-inferiority boundary</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    figures = args.output / "figures"
    figures.mkdir(parents=True)
    protocol = load(args.protocol)
    build = load(args.build_result)
    baselines = {"raw": load(args.raw_baseline), "replay": load(args.replay_baseline)}
    if build["scope"] != {"complete": True, "pages": 500, "queries": 500}:
        raise RuntimeError(f"incomplete build scope: {build['scope']}")
    for source, baseline in baselines.items():
        if baseline["slice"] != "arxivqa" or baseline["n_test_docs"] != 150 or baseline["n_eval_q"] != 150:
            raise RuntimeError(f"unexpected {source} evaluation scope")

    costs = build["timing_seconds"]["method_complete_target_build"]
    rows = []
    official_index_differences = {
        f"{source}_{ratio}": (
            float(build["storage"]["light"][source][str(ratio)]["actual_index_ndcg_at_5"])
            - quality(baselines[source], "merge_tome", ratio)
        )
        for source in ("raw", "replay")
        for ratio in (0.10, 0.05)
    }
    raw_quality = {
        "full": observed_quality(build, baselines["raw"], "raw", "full", 1.0),
        "0.1": observed_quality(build, baselines["raw"], "raw", "merge_tome", 0.10),
        "0.05": observed_quality(build, baselines["raw"], "raw", "merge_tome", 0.05),
    }
    raw_cost = {"full": costs["raw_full"], "0.1": costs["raw_light10"], "0.05": costs["raw_light05"]}
    for method_id, label, source, baseline_method, ratio in METHODS:
        measured_quality = observed_quality(
            build, baselines[source], source, baseline_method, ratio
        )
        key = "full" if ratio == 1.0 else str(ratio)
        rows.append({
            "method": method_id,
            "label": label,
            "source": source,
            "retained_ratio": ratio,
            "ndcg_at_5": measured_quality,
            "quality_delta_vs_raw": measured_quality - raw_quality[key],
            "build_seconds": float(costs[method_id]),
            "build_saving_vs_raw": 1.0 - float(costs[method_id]) / float(raw_cost[key]),
            "target_index_bytes": target_bytes(build, source, ratio),
            "persistent_reusable_bytes": int(build["storage"]["physical_ir_bytes"]) if source == "replay" else 0,
        })

    by_method = {row["method"]: row for row in rows}
    gates = {
        "scope": True,
        "dependency": bool(build["gates_before_quality"]["dependency_pass"]),
        "lowering": bool(build["gates_before_quality"]["lowering_pass"]),
        "cardinality": bool(build["gates_before_quality"]["cardinality_pass"]),
        "storage": bool(build["gates_before_quality"]["storage_pass"]),
        "actual_index_matches_official_scorer": all(
            abs(value) <= 0.0001 for value in official_index_differences.values()
        ),
        "quality_light10": by_method["reprforge_light10"]["quality_delta_vs_raw"] >= -0.01,
        "lifecycle_light10": by_method["reprforge_light10"]["build_saving_vs_raw"] >= 0.50,
    }
    gates["all_primary"] = all(gates.values())
    stats = {
        "experiment": protocol["protocol_id"],
        "primary_comparison": "reprforge_light10 versus raw_light10",
        "metric": "nDCG@5; higher is better",
        "timing_unit": "one complete 500-page target build; lower is better",
        "rows": rows,
        "gates": gates,
        "actual_index_minus_official_remerge_ndcg": official_index_differences,
        "timing_inference": "One complete transition. Batch timings are descriptive repeated stages, not independent system trials; no p-value is reported.",
        "quality_inference": "One deterministic held-out aggregate per method. No seed variance or inferential winner test is available.",
    }
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    with (args.output / "exact-table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lifecycle_svg(figures / "figure-01-matched-lifecycle-cost.svg", rows)
    quality_svg(figures / "figure-02-matched-quality-delta.svg", rows)

    primary = by_method["reprforge_light10"]
    raw_primary = by_method["raw_light10"]
    table = "\n".join(
        f"| {row['label']} | {row['ndcg_at_5']:.4f} | {row['build_seconds']:.2f} | {row['build_saving_vs_raw']*100:.2f}% | {row['target_index_bytes']/2**20:.2f} | {row['persistent_reusable_bytes']/2**20:.2f} |"
        for row in rows
    )
    report = f"""# Matched ArxivQA lifecycle analysis

## Analysis question

Under a fixed-processor ColQwen2.5 v0.1-to-v0.2 adapter/projection rollout, can dependency-valid ReprForge replay produce the same Light-compressed target-index class as raw-page rebuilding at materially lower construction cost?

## Exact table

| Method | nDCG@5 | build seconds | saving vs matched raw | target index MiB | reusable IR MiB |
|---|---:|---:|---:|---:|---:|
{table}

## Decision

Primary gate: **{'PASS' if gates['all_primary'] else 'FAIL'}**.

- At 10% retention, ReprForge changes nDCG@5 by {primary['quality_delta_vs_raw']:+.4f} relative to the exactly matched raw+Light route.
- Complete target-index construction changes from {raw_primary['build_seconds']:.2f} s to {primary['build_seconds']:.2f} s, a {primary['build_saving_vs_raw']*100:.2f}% saving.
- Both routes retain identical per-document vector counts; target-index bytes are reported separately from the persistent reusable IR.
- The nDCG values above score the actual serialized compact indexes; each is required to agree within 0.0001 with the pinned official script's independent re-merge result.

## Claim candidates

- Claim: dependency-aware replay composes with terminal geometric compression rather than competing with it.
  - Source evidence: exact-table.csv, Figure 1, Figure 2, dependency and cardinality gates.
  - Allowed wording: on this frozen transition and benchmark, ReprForge+Light produces the same target-index cardinality class with the reported quality delta and target-build saving.
  - Forbidden stronger wording: ReprForge is a better terminal compressor, or Light avoids Full target encoding by itself.
  - Uncertainty: one complete run on one RTX 4090/local-data-disk host and one adapter/projection transition.
  - Next check: repeat the frozen route on DocVQA without retuning the IR or ratio.
  - Decision: {'keep' if gates['all_primary'] else 'weaken'}.

## Boundaries

The quality rows are deterministic aggregates over the pinned 150-document/150-query holdout. The experiment does not provide independent quality seeds. The build measurement is one complete transition; batch-level values describe within-run stage heterogeneity and are not treated as independent trials.

The v0.1 and v0.2 repositories ship different processor defaults. Both measured routes intentionally pin the hashed v0.2 processor, so this is an adapter-only rollout result, not a claim that a complete v0.1 package can reuse its prefix after accepting every v0.2 package change.
"""
    (args.output / "analysis-report.md").write_text(report)
    (args.output / "stats-appendix.md").write_text(
        """# Statistical appendix

- Unit of quality analysis: one query in the frozen document-disjoint holdout; the saved official baseline JSON exposes only the deterministic aggregate, so no query-level confidence interval is reconstructed.
- Unit of system analysis: one complete 500-page transition. Batch timings are dependent stages within that run, not independent replicates.
- Consequently no t-test, Wilcoxon test, or p-value is reported. The preregistered decisions are engineering/quality gates on exact aggregate differences.
- Physical-index fidelity is checked by comparing direct scoring of each serialized Light index with the pinned official script's independent merge from its BF16 Full cache; the maximum permitted absolute nDCG difference is 0.0001.
- The 10% comparison is primary. The 5% comparison is a frozen secondary stress boundary and is not used to switch the main decision.
"""
    )
    (args.output / "figure-catalog.md").write_text(
        """# Figure catalog

## figures/figure-01-matched-lifecycle-cost.svg

- Purpose: compare complete target-build cost after matching the final index operator and retained-vector ratio.
- Data: one complete raw and ReprForge route for Full, Light 10%, and Light 5%.
- Notice: whether replay remains materially cheaper after charging the identical terminal merge and serialization.
- Implication: separates lifecycle reuse from terminal compression.
- Caveat: normalized bars are one complete run, not a multi-host confidence interval.

## figures/figure-02-matched-quality-delta.svg

- Purpose: test whether replay remains quality-noninferior after the same terminal compression.
- Data: deterministic nDCG@5 difference between each ReprForge route and its matched raw route.
- Notice: the primary Light-10% point relative to the frozen -0.01 boundary.
- Implication: determines whether the lifecycle payoff survives at the actual serving-index budget.
- Caveat: no error bars because the official output contains one deterministic aggregate per condition.
"""
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
