#!/usr/bin/env python3
"""Run deterministic bounded-arrival replay on frozen HR/Finance traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reprforge.candidate_fusion import _candidate_ndcg
from reprforge.intervention_utility import _ndcg_row
from reprforge.progressive_oracle import load_trace, rank_order, validate_pair
from reprforge.windowed_arrival_replay import (
    POLICIES,
    make_arrival_times,
    replay_windowed_arrivals,
)


DEFAULT_WINDOWS = (1, 8, 16, 32, 64)
DEFAULT_SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_domain(root: Path, candidate_k: int) -> dict:
    manifest_layout = (root / "bm25" / "manifest.json").is_file()
    if manifest_layout:
        text_root = root / "bm25"
        visual_root = root / "visual"
        text_trace = load_trace(text_root)
        visual_trace = load_trace(visual_root)
        qrels = validate_pair(text_trace, visual_trace)
        text_scores = text_trace.scores
        visual_scores = visual_trace.scores
        corpus_ids = text_trace.corpus_ids
        text = text_trace
        visual = visual_trace
        text_path = text_root / text_trace.manifest["runtime_file"]
        visual_path = visual_root / visual_trace.manifest["runtime_file"]
        labels_path = text_root / text_trace.manifest["oracle_labels_file"]
        manifest_provenance = {
            "text_manifest": {
                "path": str((text_root / "manifest.json").resolve()),
                "sha256": sha256(text_root / "manifest.json"),
            },
            "visual_manifest": {
                "path": str((visual_root / "manifest.json").resolve()),
                "sha256": sha256(visual_root / "manifest.json"),
            },
        }
    else:
        text_path = root / "text-runtime.npz"
        visual_path = root / "visual-runtime.npz"
        labels_path = root / "oracle-labels.npz"
        for path in (text_path, visual_path, labels_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(text_path, allow_pickle=False) as payload:
            text_values = {name: payload[name] for name in payload.files}
        with np.load(visual_path, allow_pickle=False) as payload:
            visual_values = {name: payload[name] for name in payload.files}
        with np.load(labels_path, allow_pickle=False) as payload:
            labels = {name: payload[name] for name in payload.files}
        if not np.array_equal(text_values["query_ids"], visual_values["query_ids"]):
            raise ValueError(f"query identifiers differ in {root}")
        if not np.array_equal(text_values["corpus_ids"], visual_values["corpus_ids"]):
            raise ValueError(f"corpus identifiers differ in {root}")
        text_scores = np.asarray(text_values["scores"], dtype=np.float64)
        visual_scores = np.asarray(visual_values["scores"], dtype=np.float64)
        if text_scores.shape != visual_scores.shape:
            raise ValueError(f"text/visual score shape mismatch in {root}")
        qrels = np.zeros(text_scores.shape, dtype=np.int16)
        qrels[
            np.asarray(labels["query_positions"], dtype=np.int64),
            np.asarray(labels["corpus_positions"], dtype=np.int64),
        ] = np.asarray(labels["relevance"], dtype=np.int16)
        if np.any(qrels.max(axis=1) == 0):
            raise ValueError(f"at least one query has no qrel in {root}")
        corpus_ids = np.asarray(text_values["corpus_ids"])
        text = SimpleNamespace(scores=text_scores, corpus_ids=corpus_ids)
        visual = SimpleNamespace(scores=visual_scores, corpus_ids=corpus_ids)
        manifest_provenance = {}
    text_order = rank_order(text_scores, corpus_ids)
    candidate_k = min(candidate_k, text_scores.shape[1])
    base_quality = np.asarray(
        [
            _ndcg_row(text_scores[q], qrels[q], corpus_ids, cutoff=10)
            for q in range(text_scores.shape[0])
        ],
        dtype=np.float64,
    )
    refined_quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        text_order,
        candidate_k=candidate_k,
        method="zscore_sum",
        cutoff=10,
    )
    return {
        "cohorts": text_order[:, :candidate_k].tolist(),
        "quality_gain": refined_quality - base_quality,
        "base_mean_quality": float(base_quality.mean()),
        "refined_mean_quality": float(refined_quality.mean()),
        "query_count": int(text_scores.shape[0]),
        "corpus_pages": int(text_scores.shape[1]),
        "candidate_union_pages": int(
            len(set(text_order[:, :candidate_k].reshape(-1).tolist()))
        ),
        "trace_layout": "manifest-pair" if manifest_layout else "legacy-flat",
        "provenance": {
            "root": str(root.resolve()),
            "text_runtime": {
                "path": str(text_path.resolve()),
                "sha256": sha256(text_path),
                "bytes": text_path.stat().st_size,
            },
            "visual_runtime": {
                "path": str(visual_path.resolve()),
                "sha256": sha256(visual_path),
                "bytes": visual_path.stat().st_size,
            },
            "oracle_labels": {
                "path": str(labels_path.resolve()),
                "sha256": sha256(labels_path),
                "bytes": labels_path.stat().st_size,
                "use": "post-hoc quality-work accounting only",
            },
            **manifest_provenance,
        },
    }


def aggregate_runs(runs: list[dict]) -> dict:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in runs:
        grouped.setdefault(
            (row["arrival_model"], row["window"], row["policy"]), []
        ).append(row)
    result: dict[str, dict] = {}
    for (model, window, policy), rows in sorted(grouped.items()):
        completion = np.concatenate(
            [np.asarray(row["raw_completion_pages"], dtype=np.float64) for row in rows]
        )
        waits = np.concatenate(
            [np.asarray(row["raw_wait_work"], dtype=np.float64) for row in rows]
        )
        bypass = np.concatenate(
            [np.asarray(row["raw_bypass_count"], dtype=np.int64) for row in rows]
        )
        auc = np.asarray([row["quality_work_auc"] for row in rows])
        starvation = bypass >= window
        model_result = result.setdefault(model, {})
        window_result = model_result.setdefault(str(window), {})
        window_result[policy] = {
            "runs": len(rows),
            "completion_pages": {
                "mean": float(completion.mean()),
                "p50": float(np.quantile(completion, 0.50)),
                "p95": float(np.quantile(completion, 0.95)),
                "max": float(completion.max()),
            },
            "quality_work_auc": {
                "mean": float(auc.mean()),
                "p50": float(np.quantile(auc, 0.50)),
                "min": float(auc.min()),
                "max": float(auc.max()),
            },
            "wait_work": {
                "mean": float(waits.mean()),
                "p50": float(np.quantile(waits, 0.50)),
                "p95": float(np.quantile(waits, 0.95)),
                "max": float(waits.max()),
            },
            "starvation": {
                "count": int(starvation.sum()),
                "fraction": float(starvation.mean()),
                "max_younger_bypass": int(bypass.max()),
            },
        }
    return result


def assess_window(
    datasets: dict[str, dict],
    window: int,
    arrival_models: tuple[str, ...],
    *,
    policy: str,
) -> dict:
    checks = []
    for dataset_name, dataset in datasets.items():
        aggregate = dataset["aggregate"]
        full_window = str(dataset["query_count"])
        for model in arrival_models:
            fifo = aggregate[model]["1"]["fifo"]
            bounded = aggregate[model][str(window)][policy]
            full = aggregate[model][full_window][policy]
            fifo_completion = fifo["completion_pages"]["mean"]
            full_gain = fifo_completion - full["completion_pages"]["mean"]
            bounded_gain = fifo_completion - bounded["completion_pages"]["mean"]
            preservation = bounded_gain / full_gain if full_gain > 0 else None
            auc_delta = (
                bounded["quality_work_auc"]["mean"]
                - fifo["quality_work_auc"]["mean"]
            )
            passes = bool(
                preservation is not None
                and preservation >= 0.5
                and auc_delta >= -1e-12
            )
            checks.append(
                {
                    "dataset": dataset_name,
                    "arrival_model": model,
                    "fifo_mean_completion_pages": fifo_completion,
                    "full_pending_frontier_gain_pages": full_gain,
                    "bounded_frontier_gain_pages": bounded_gain,
                    "full_gain_preserved_fraction": preservation,
                    "quality_work_auc_delta_vs_fifo": auc_delta,
                    "passes": passes,
                }
            )
    return {
        "window": window,
        "policy": policy,
        "criterion": (
            "preserve >=50% of full-pending frontier completion-work gain and "
            "mean quality-work AUC >= FIFO for both datasets and arrival models"
        ),
        "checks": checks,
        "passes": all(row["passes"] for row in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--burst-size", type=int, default=32)
    parser.add_argument("--burst-interval", type=float, default=64.0)
    parser.add_argument("--poisson-mean", type=float, default=4.0)
    args = parser.parse_args()
    if len(args.seeds) < 5:
        raise ValueError("at least five deterministic permutation seeds are required")
    required_windows = {1, 8, 16, 32, 64}
    if not required_windows.issubset(args.windows):
        raise ValueError("windows must include 1, 8, 16, 32, and 64")

    dataset_roots = {"hr": args.data_root / "hr", "finance": args.data_root / "finance"}
    datasets: dict[str, dict] = {}
    arrival_models = ("burst", "poisson")
    for dataset_name, root in dataset_roots.items():
        loaded = load_domain(root, args.candidate_k)
        windows = sorted(set(args.windows) | {loaded["query_count"]})
        runs: list[dict] = []
        for seed in args.seeds:
            arrival_order = np.random.default_rng(seed).permutation(
                loaded["query_count"]
            )
            for model_offset, model in enumerate(arrival_models):
                arrival_times = make_arrival_times(
                    loaded["query_count"],
                    model=model,
                    seed=seed + 10000 * model_offset,
                    burst_size=args.burst_size,
                    burst_interval=args.burst_interval,
                    poisson_mean=args.poisson_mean,
                )
                for window in windows:
                    for policy_offset, policy in enumerate(POLICIES):
                        replay = replay_windowed_arrivals(
                            loaded["cohorts"],
                            arrival_order,
                            arrival_times,
                            loaded["quality_gain"],
                            base_mean_quality=loaded["base_mean_quality"],
                            corpus_pages=loaded["corpus_pages"],
                            batch_size=args.batch_size,
                            window=window,
                            policy=policy,
                            random_seed=seed + 1000 * policy_offset,
                        )
                        row = replay.as_dict()
                        row.update(
                            {
                                "permutation_seed": seed,
                                "arrival_model": model,
                                "raw_completion_pages": replay.completion_pages,
                                "raw_wait_work": replay.wait_work,
                                "raw_bypass_count": replay.bypass_count,
                            }
                        )
                        runs.append(row)
        datasets[dataset_name] = {
            key: value
            for key, value in loaded.items()
            if key not in {"cohorts", "quality_gain"}
        }
        datasets[dataset_name]["aggregate"] = aggregate_runs(runs)
        # Aggregate tails are computed from query-level arrays above.  The JSON
        # keeps compact per-run summaries rather than duplicating those arrays.
        datasets[dataset_name]["runs"] = [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("raw_")
            }
            for row in runs
        ]

    gate_16 = assess_window(
        datasets, 16, arrival_models, policy="frontier"
    )
    gate_32 = assess_window(
        datasets, 32, arrival_models, policy="frontier"
    )
    gate_64 = assess_window(
        datasets, 64, arrival_models, policy="frontier"
    )
    fair_gate_16 = assess_window(
        datasets, 16, arrival_models, policy="frontier_fair"
    )
    fair_gate_32 = assess_window(
        datasets, 32, arrival_models, policy="frontier_fair"
    )
    gate_passes = gate_16["passes"] or gate_32["passes"]
    current_manifest_traces = all(
        value["trace_layout"] == "manifest-pair" for value in datasets.values()
    )
    evidence_scope = (
        {
            "status": "current-manifest-frozen-replay",
            "warning": None,
        }
        if current_manifest_traces
        else {
            "status": "legacy-provenance-stress-test",
            "warning": (
                "These flat heterogeneity-atlas traces are not the current "
                "895-page HR / 1855-page Finance cohort-compiler traces."
            ),
        }
    )
    report = {
        "schema_version": 1,
        "stage": "frozen-causal-windowed-arrival-replay",
        "evidence_scope": evidence_scope,
        "candidate_k": args.candidate_k,
        "request_batch_size": args.batch_size,
        "windows": sorted(set(args.windows)),
        "full_pending_window_added_per_dataset": True,
        "permutation_seeds": list(args.seeds),
        "arrival_models": {
            "burst": {
                "burst_size": args.burst_size,
                "burst_interval_page_work": args.burst_interval,
            },
            "poisson": {"mean_interarrival_page_work": args.poisson_mean},
        },
        "observation_contract": {
            "scheduler_can_observe": [
                "candidate-page membership of arrived queries in the oldest W pending requests",
                "resident and current staged page identifiers",
                "candidate membership of past arrivals for the history baseline",
            ],
            "scheduler_cannot_observe": [
                "future arrivals",
                "relevance labels",
                "visual scores",
                "per-query quality gain",
            ],
            "qrels_use": "post-hoc frozen nDCG@10 quality-work accounting only",
        },
        "metrics": {
            "completion_work": "unique encoded pages resident at atomic query publication",
            "quality_work_auc": (
                "left-continuous mean nDCG@10 integrated over encoded pages, "
                "normalized by full corpus pages"
            ),
            "wait_work": "page-work clock at batch start minus query arrival clock",
            "starvation": (
                "query has at least W younger arrivals dispatched before it; "
                "max_younger_bypass is also reported"
            ),
        },
        "datasets": datasets,
        "gate": {
            "window_16": gate_16,
            "window_32": gate_32,
            "window_64_diagnostic": gate_64,
            "hard_fairness_diagnostic": {
                "definition": "force service before W younger arrivals can bypass a query",
                "window_16": fair_gate_16,
                "window_32": fair_gate_32,
            },
            (
                "current_frontier_claim_decision"
                if current_manifest_traces
                else "legacy_stress_test_decision"
            ): "GO" if gate_passes else "NO-GO",
            "decision": "GO" if current_manifest_traces and gate_passes else "NO-GO",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate": report["gate"],
                "dataset_quality": {
                    name: {
                        "base_mean_ndcg@10": value["base_mean_quality"],
                        "refined_mean_ndcg@10": value["refined_mean_quality"],
                    }
                    for name, value in datasets.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
