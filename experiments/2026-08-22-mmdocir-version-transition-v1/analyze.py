#!/usr/bin/env python3
"""Create the strict analysis bundle for the complete version transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("result.json"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output"))
    return parser.parse_args()


def svg_timeline(result: dict[str, Any]) -> str:
    timing = result["timing"]
    replay = timing["replay"]
    raw = float(timing["raw_rebuild_transition_seconds"])
    warm = float(timing["warm_transition_seconds"])
    scale = 600 / raw
    shared = (
        float(replay["query_encoding_seconds"])
        + float(replay["quality_validation_seconds"])
        + float(timing["serving_materialization_seconds"])
        + float(timing["generation_hash_seconds"])
        + float(timing["publication_seconds"])
    )
    cold_build = raw - shared
    warm_build = warm - shared
    rows = [
        ("Raw-page rebuild", cold_build, shared, "#667085"),
        ("ReprForge IR replay", warm_build, shared, "#2563eb"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="300" viewBox="0 0 980 300">',
        '<rect width="980" height="300" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#101828}.label{font-size:16px}.value{font-size:15px;font-weight:bold}.small{font-size:13px;fill:#475467}</style>',
        '<text x="24" y="32" font-size="20" font-weight="bold">Complete adapter-version transition wall time</text>',
        '<text x="24" y="54" class="small">Build new terminal → validate quality → build SQ8 → hash → publish</text>',
    ]
    for index, (label, build, common, color) in enumerate(rows):
        y = 92 + index * 82
        parts.append(f'<text x="24" y="{y + 23}" class="label">{label}</text>')
        x = 210
        parts.append(f'<rect x="{x}" y="{y}" width="{build * scale:.2f}" height="34" rx="4" fill="{color}"/>')
        parts.append(f'<rect x="{x + build * scale:.2f}" y="{y}" width="{common * scale:.2f}" height="34" rx="4" fill="#f59e0b"/>')
        parts.append(f'<text x="{x + (build + common) * scale + 10:.2f}" y="{y + 23}" class="value">{build + common:.1f} s</text>')
    parts.extend([
        '<rect x="210" y="255" width="16" height="12" fill="#2563eb"/><text x="234" y="266" class="small">representation build</text>',
        '<rect x="390" y="255" width="16" height="12" fill="#f59e0b"/><text x="414" y="266" class="small">shared validation, serving build, hash, publication</text>',
        '</svg>',
    ])
    return "".join(parts)


def svg_storage(result: dict[str, Any]) -> str:
    storage = result["storage"]
    terminal = float(storage["old_terminal_bytes"])
    values = [
        ("Current terminal", terminal),
        ("Reusable IR", float(storage["physical_ir_bytes"])),
        ("New terminal", float(storage["new_terminal_bytes"])),
        ("Disposable SQ8", float(storage["new_sq8_serving_bytes"])),
    ]
    max_value = max(value for _, value in values)
    scale = 430 / max_value
    colors = ["#667085", "#7c3aed", "#2563eb", "#f59e0b"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="390" viewBox="0 0 940 390">',
        '<rect width="940" height="390" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#101828}.label{font-size:16px}.value{font-size:14px;font-weight:bold}.small{font-size:13px;fill:#475467}</style>',
        '<text x="24" y="32" font-size="20" font-weight="bold">Physical storage roles are distinct</text>',
        '<text x="24" y="54" class="small">IR is optional rebuild state; terminal vectors are exact-rerank state; SQ8 is disposable candidate state.</text>',
    ]
    for index, ((label, value), color) in enumerate(zip(values, colors)):
        y = 84 + index * 62
        parts.append(f'<text x="24" y="{y + 23}" class="label">{label}</text>')
        parts.append(f'<rect x="190" y="{y}" width="{value * scale:.2f}" height="32" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{200 + value * scale:.2f}" y="{y + 22}" class="value">{value / 1e9:.3f} GB · {value / terminal:.3f}×</text>')
    peak = float(storage["transition_peak_lower_bound_bytes"])
    steady = float(storage["steady_state_ir_plus_generation_bytes"])
    parts.append(f'<text x="24" y="344" class="small">Steady IR + new generation: {steady / 1e9:.3f} GB. Transition peak lower bound with old terminal retained: {peak / 1e9:.3f} GB.</text>')
    parts.append('</svg>')
    return "".join(parts)


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text())
    protocol = json.loads(args.protocol.read_text())
    output = args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "figure-01-transition-timeline.svg").write_text(svg_timeline(result))
    (figures / "figure-02-storage-roles.svg").write_text(svg_storage(result))

    quality = result["quality"]
    recall = quality["pca256"]["recall_at_5"]
    full_recall = quality["full"]["recall_at_5"]
    ndcg = quality["pca256"]["ndcg_at_10"]
    full_ndcg = quality["full"]["ndcg_at_10"]
    interval = result["timing"]["replay"]
    replay_result = json.loads(Path(args.result.parent / "replay-result.json").read_text())
    recall_ci = replay_result["paired_pca_minus_full_recall_at_5"]["bootstrap_95"]
    saving = result["timing"]["end_to_end_transition_saving"]
    candidate = result["candidate_fidelity"]
    gates = result["gates"]
    gate_rows = "\n".join(f"- `{name}`: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    report = f"""# MMDocIR complete version-transition analysis

## Decision

{'**PROMOTE**' if result['all_gates_pass'] else '**DO NOT PROMOTE**'} the dependency-aware version transition under the frozen contract. The experiment {'completed' if result['all_gates_pass'] else 'did not complete'} the physical path from a post-vision IR to a quality-admitted, SQ8-indexed, atomically published target generation.

## What was tested

The target change updates the ColQwen2.5 retrieval decoder adapter and projection while preserving the processor, base embedding, and vision dependencies compiled into the post-vision IR. All {result['scope']['documents']} MMDocIR documents, {result['scope']['pages']:,} pages, and {result['scope']['vectors']:,} terminal vectors were regenerated. Official quality uses {protocol['benchmark']['queries']:,} source-document queries; serving fidelity uses {result['scope']['global_probe_queries']} domain-stratified queries over all pages.

This is an execution of one method, not a sum of unrelated best numbers: replay writes a new terminal generation; that exact generation is scanned to train and populate SQ8; its files are hashed into one manifest; publication occurs only after the quality, candidate, scope, and timing gates pass.

## Decisive results

- Official Recall@5 is {recall:.6f} versus Full {full_recall:.6f}; paired compact-minus-Full 95% query-bootstrap interval is [{recall_ci[0]:+.6f}, {recall_ci[1]:+.6f}].
- Official nDCG@10 is {ndcg:.6f} versus Full {full_ndcg:.6f} (delta {ndcg - full_ndcg:+.6f}). These are quality-preservation results, not improvements.
- The complete warm transition takes {result['timing']['warm_transition_seconds']:.2f} s versus {result['timing']['raw_rebuild_transition_seconds']:.2f} s for the raw-page route with identical downstream validation and serving work: {saving:.2%} saving.
- SQ8 returns {candidate['mean_candidate_pages']:.1f} candidate pages on average ({candidate['mean_candidate_corpus_fraction']:.2%} of the collection) and covers {candidate['mean_full_exact_top10_candidate_recall']:.2%} of Full exact global Top-10 pages. Median candidate search is {candidate['search_latency_ms']['median']:.2f} ms.
- The new generation was {'published' if result['publication']['published'] else 'not published'} through a single manifest-addressed `ACTIVE` pointer. This run tests first publication into a new deployment root; the separate five-trial cutover experiment is the evidence for concurrent old/new switching.

## Gate audit

{gate_rows}

## Mechanism interpretation

The payoff comes from dependency scope, not generic feature compression. The adapter/projection update invalidates terminal vectors but does not invalidate the frozen visual prefix. Replaying the deepest valid artifact avoids the vision path while still materializing the true target-version endpoint. A query bridge is cheaper but serves stale document semantics and was already significantly worse on the same complete benchmark. A terminal rewrite is faster but is legal only when embedding semantics do not change.

The lossy in-flight merge is not required for this result. Its one-shot 7% build saving and the failed ViDoSeek query-free risk certificate make it an optional workload-admitted operator, not the default ReprForge method.

## What this can support

1. A problem contribution: multimodal multi-vector indexes require dependency-aware version maintenance, not unconditional re-embedding.
2. A method contribution: fingerprinted dependency cuts plus measured materialization planning choose the deepest quality-valid rebuild source.
3. A systems contribution: the selected route produces and admits one immutable serving generation, including terminal replay, compact candidate construction, validation, manifest hashing, and publication.
4. A mechanism result: embedding-space alignment is not a MaxSim compatibility certificate, and query-free merge geometry is not a ranking-safety certificate.

PCA, SQ8, atomic rename, and DAG caching are components and baselines; none is claimed as standalone novelty.

## Limits

- One complete adapter/projection transition, one target backbone, one local A100/NVMe environment.
- MMDocIR official evaluation is source-document page ranking; the global candidate probe is internal serving fidelity.
- The reported transition peak is a lower bound because it excludes filesystem and process overhead and any separately retained v0.1 ANN.
- Distributed/object-store publication, garbage collection, cold-cache performance, and production update-frequency traces remain outside this experiment.
"""
    (output / "analysis-report.md").write_text(report)
    stats = f"""# Statistical appendix

- Unit for official quality inference: query (`n={protocol['benchmark']['queries']}`).
- Recall@5 contrast: {recall - full_recall:+.9f}; 20,000-draw paired query bootstrap 95% interval [{recall_ci[0]:+.9f}, {recall_ci[1]:+.9f}].
- nDCG@10 contrast: {ndcg - full_ndcg:+.9f}; descriptive here because the frozen primary inference gate was Recall@5.
- Global candidate probe: `n={result['scope']['global_probe_queries']}` deterministic, domain-stratified queries. Mean Full exact Top-10 candidate recall {candidate['mean_full_exact_top10_candidate_recall']:.6f}. It is not an official benchmark estimate.
- Timing is one complete execution on a shared server. The comparison holds downstream serving/validation work constant and replaces only raw-page construction with IR replay. It is not a hardware-general confidence interval.
- All primary gates were frozen in `protocol.json` before full execution; no hyperparameter or threshold sweep followed observation of this result.
"""
    (output / "stats-appendix.md").write_text(stats)
    (output / "figure-catalog.md").write_text(
        "# Figure catalog\n\n"
        "- `figure-01-transition-timeline.svg`: fair end-to-end raw versus IR transition; shared downstream phases appear in both bars.\n"
        "- `figure-02-storage-roles.svg`: separates optional reusable IR, exact terminal page store, and disposable SQ8 candidate state.\n"
    )
    summary = {
        "all_gates_pass": result["all_gates_pass"],
        "recall_at_5": {"full": full_recall, "reprforge": recall, "bootstrap_95": recall_ci},
        "ndcg_at_10": {"full": full_ndcg, "reprforge": ndcg},
        "end_to_end_transition_saving": saving,
        "candidate_recall": candidate["mean_full_exact_top10_candidate_recall"],
        "published": result["publication"]["published"],
    }
    (output / "analysis-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
