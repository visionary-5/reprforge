#!/usr/bin/env python3
"""Produce a deterministic strict-analysis summary and two dependency-free SVGs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "gpu-result.json"
OUTPUT = ROOT / "analysis-output"
FIGURES = OUTPUT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def text(x: float, y: float, value: object, size: int = 16, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#222">{esc(value)}</text>'


def timing_figure(data: dict) -> str:
    raw = data["observations"]["raw"]
    replay = data["observations"]["replay"]
    width, height = 900, 520
    left, top, chart_w, chart_h = 105, 80, 700, 320
    maximum = 40.0
    colors = {"source": "#56B4E9", "suffix": "#E69F00"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 34, "Official ColQwen2 v0.1 → v1.0 rebuild time", 22, "middle", 700),
        text(width / 2, 58, "128 Energy pages, A100, warm same-process repetitions", 14, "middle"),
    ]
    for tick in range(0, 41, 10):
        y = top + chart_h - tick / maximum * chart_h
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        parts.append(text(left - 12, y + 5, tick, 14, "end"))
    parts.append(text(25, top + chart_h / 2, "seconds", 15, "middle"))
    bar_w, gap = 64, 34
    all_rows = [("Raw", row) for row in raw] + [("Replay", row) for row in replay]
    for i, (method, row) in enumerate(all_rows):
        x = left + 38 + i * (bar_w + gap)
        source_h = row["source_seconds"] / maximum * chart_h
        suffix_h = row["decoder_projection_d2h_seconds"] / maximum * chart_h
        base_y = top + chart_h
        parts.append(f'<rect x="{x}" y="{base_y - source_h}" width="{bar_w}" height="{source_h}" fill="{colors["source"]}"/>')
        parts.append(f'<rect x="{x}" y="{base_y - source_h - suffix_h}" width="{bar_w}" height="{suffix_h}" fill="{colors["suffix"]}"/>')
        parts.append(text(x + bar_w / 2, base_y + 22, f"{method[0]}{row['repeat'] + 1}", 13, "middle"))
        parts.append(text(x + bar_w / 2, base_y - source_h - suffix_h - 8, f"{row['wall_seconds']:.2f}", 13, "middle", 700))
    parts.extend([
        f'<rect x="{left + 420}" y="{height - 66}" width="18" height="18" fill="{colors["source"]}"/>',
        text(left + 445, height - 52, "image/processor/vision or IR load", 14),
        f'<rect x="{left + 420}" y="{height - 39}" width="18" height="18" fill="{colors["suffix"]}"/>',
        text(left + 445, height - 25, "target decoder + projection + D2H", 14),
        text(left, height - 34, "R = raw; P = replay", 14),
        "</svg>",
    ])
    return "\n".join(parts) + "\n"


def storage_figure(data: dict) -> str:
    ir = data["source_ir"]
    entries = [
        ("Compressed images", 1.0, "#999999"),
        ("Target terminal", ir["ir_to_image_ratio"] / ir["ir_to_terminal_ratio"], "#009E73"),
        ("Exact post-vision IR", ir["ir_to_image_ratio"], "#D55E00"),
    ]
    width, height = 900, 430
    left, top, chart_w, chart_h = 240, 70, 560, 250
    maximum = 5.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 34, "Exact replay is fast but the uncompressed IR is not storage-admissible", 21, "middle", 700),
    ]
    for tick in range(0, 6):
        x = left + tick / maximum * chart_w
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + chart_h}" stroke="#ddd"/>')
        parts.append(text(x, top + chart_h + 25, f"{tick}×", 13, "middle"))
    for i, (label, ratio, color) in enumerate(entries):
        y = top + 28 + i * 74
        bar = ratio / maximum * chart_w
        parts.append(text(left - 15, y + 23, label, 15, "end"))
        parts.append(f'<rect x="{left}" y="{y}" width="{bar}" height="34" fill="{color}"/>')
        parts.append(text(left + bar + 10, y + 23, f"{ratio:.2f}× images", 14, "start", 700))
    parts.append(text(width / 2, height - 45, f"IR is {ir['ir_to_terminal_ratio']:.2f}× the float32 terminal tensor; storage is a diagnostic failure, not a claimed win.", 14, "middle"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    data = json.loads(RESULT.read_text())
    if not data["all_gates_pass"]:
        raise RuntimeError("GPU protocol did not pass all frozen gates")
    raw = [row["wall_seconds"] for row in data["observations"]["raw"]]
    replay = [row["wall_seconds"] for row in data["observations"]["replay"]]
    paired = [a - b for a, b in zip(raw, replay)]
    paired_fraction = [1.0 - b / a for a, b in zip(raw, replay)]
    mean_delta = statistics.fmean(paired)
    sd_delta = statistics.stdev(paired)
    t_critical_df2 = 4.302652729911275
    half_width = t_critical_df2 * sd_delta / math.sqrt(len(paired))
    summary = {
        "gpu_result_sha256": sha256(RESULT),
        "unit_of_analysis": "three warm same-workload technical repetitions; not independent datasets, seeds, or hardware replicas",
        "n_pairs": len(paired),
        "raw_seconds": raw,
        "replay_seconds": replay,
        "paired_absolute_saving_seconds": paired,
        "paired_fractional_saving": paired_fraction,
        "mean_absolute_saving_seconds": mean_delta,
        "sd_absolute_saving_seconds": sd_delta,
        "descriptive_t_interval_95_seconds": [mean_delta - half_width, mean_delta + half_width],
        "median_speedup_ratio": statistics.median(raw) / statistics.median(replay),
        "minimum_fractional_saving": min(paired_fraction),
        "paired_wins": sum(value > 0 for value in paired),
        "exact_two_sided_sign_test_p": 0.25,
        "inference_boundary": "n=3 makes normality tests and broad inferential claims invalid; the t interval is descriptive for warm-run variability only, and the sign test is underpowered",
        "target_equivalence": data["target_equivalence"],
        "version_invalidation": data["version_invalidation"],
        "storage": data["source_ir"],
    }
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (OUTPUT / "stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (FIGURES / "figure-01-rebuild-time.svg").write_text(timing_figure(data))
    (FIGURES / "figure-02-storage-boundary.svg").write_text(storage_figure(data))


if __name__ == "__main__":
    main()
