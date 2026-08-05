#!/usr/bin/env python3
"""Cross-fit query-adaptive Omni cascade policies without test-fold leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.compression_risk_metrics import (
    bootstrap_mean_interval,
    regret_summary,
)
from tools.analyze_omni_pair import _query_metrics, load_qrels


PLANS = (20, 50, 100, 1110)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scored_ranking(
    path: Path, *, expected_depth: int
) -> dict[str, list[tuple[str, float]]]:
    output: dict[str, list[tuple[str, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            query_id, doc_id, score_text = fields
            output.setdefault(query_id, []).append((doc_id, float(score_text)))
    if any(len(rows) != expected_depth for rows in output.values()):
        raise ValueError(f"unexpected ranking depth in {path}")
    if any(len({doc_id for doc_id, _ in rows}) != len(rows) for rows in output.values()):
        raise ValueError(f"duplicate document in {path}")
    return output


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int = 3000,
    learning_rate: float = 0.05,
    l2: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("logistic inputs must be aligned and non-empty")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    weights = np.zeros(normalized.shape[1] + 1, dtype=np.float64)
    positive = float(y.mean())
    if positive == 0.0 or positive == 1.0:
        weights[0] = 30.0 if positive == 1.0 else -30.0
        return mean, scale, weights
    sample_weights = np.where(
        y > 0.0,
        0.5 / positive,
        0.5 / (1.0 - positive),
    )
    for _ in range(steps):
        probabilities = _sigmoid(weights[0] + normalized @ weights[1:])
        error = sample_weights * (probabilities - y)
        weights[0] -= learning_rate * float(error.mean())
        weights[1:] -= learning_rate * (
            normalized.T @ error / len(y) + l2 * weights[1:]
        )
    return mean, scale, weights


def predict_logistic(
    model: tuple[np.ndarray, np.ndarray, np.ndarray], features: np.ndarray
) -> np.ndarray:
    mean, scale, weights = model
    normalized = (np.asarray(features, dtype=np.float64) - mean) / scale
    return _sigmoid(weights[0] + normalized @ weights[1:])


def admission_threshold(
    scores: np.ndarray,
    unsafe: np.ndarray,
    *,
    maximum_empirical_risk: float,
    minimum_admissions: int = 10,
) -> float:
    if not 0.0 <= maximum_empirical_risk <= 1.0:
        raise ValueError("empirical risk must lie in [0, 1]")
    order = np.argsort(scores, kind="stable")
    best = 0
    for count in range(minimum_admissions, len(order) + 1):
        if float(np.mean(unsafe[order[:count]])) <= maximum_empirical_risk:
            best = count
    if best == 0:
        return -1.0
    return float(scores[order[best - 1]] + 1e-12)


def _query_features(
    hpool: list[tuple[str, float]],
    query_text: str,
    *,
    agc: list[tuple[str, float]] | None,
) -> tuple[list[str], np.ndarray]:
    scores = np.asarray([score for _, score in hpool], dtype=np.float64)
    standard = (scores - scores.mean()) / (scores.std() + 1e-9)
    names = ["query_characters", "query_words", "question_marks", "commas"]
    values = [
        len(query_text) / 200.0,
        len(query_text.split()) / 30.0,
        float(query_text.count("?")),
        float(query_text.count(",")),
    ]
    for depth in (1, 2, 5, 10, 20, 50):
        names.append(f"hpool_standard_score_at_{depth}")
        values.append(float(standard[depth - 1]))
    for depth in (1, 5, 10, 20, 50):
        names.append(f"hpool_standard_margin_at_{depth}")
        values.append(float((scores[depth - 1] - scores[depth]) / (scores.std() + 1e-9)))
    for depth in (5, 10, 20, 50, 100):
        names.extend((f"hpool_prefix_mean_at_{depth}", f"hpool_prefix_std_at_{depth}"))
        values.extend((float(standard[:depth].mean()), float(standard[:depth].std())))
    if agc is not None:
        hpool_docs = [doc_id for doc_id, _ in hpool]
        agc_docs = [doc_id for doc_id, _ in agc]
        agc_positions = {doc_id: rank for rank, doc_id in enumerate(agc_docs, start=1)}
        for depth in (1, 5, 10, 20, 50, 100):
            names.append(f"hpool_agc_overlap_at_{depth}")
            values.append(
                len(set(hpool_docs[:depth]) & set(agc_docs[:depth])) / depth
            )
        for depth in (5, 10, 20):
            names.append(f"hpool_prefix_mean_agc_rank_at_{depth}")
            values.append(
                float(
                    np.mean(
                        [min(agc_positions.get(doc_id, 101), 101) for doc_id in hpool_docs[:depth]]
                    )
                    / 100.0
                )
            )
    return names, np.asarray(values, dtype=np.float64)


def _progressive_features(
    hpool: list[tuple[str, float]],
    cascade: list[tuple[str, float]],
) -> tuple[list[str], np.ndarray]:
    depth = len(cascade)
    hpool_prefix = hpool[:depth]
    hpool_docs = [doc_id for doc_id, _ in hpool_prefix]
    cascade_docs = [doc_id for doc_id, _ in cascade]
    hpool_scores = dict(hpool_prefix)
    cascade_scores = dict(cascade)
    cascade_values = np.asarray([score for _, score in cascade], dtype=np.float64)
    scale = cascade_values.std() + 1e-9
    standard = (cascade_values - cascade_values.mean()) / scale
    cascade_positions = {
        doc_id: position for position, doc_id in enumerate(cascade_docs)
    }
    correlation = float(
        np.corrcoef(
            [hpool_scores[doc_id] for doc_id in hpool_docs],
            [cascade_scores[doc_id] for doc_id in hpool_docs],
        )[0, 1]
    )
    names = [
        f"stage_{depth}_same_top1",
        f"stage_{depth}_top10_overlap",
        f"stage_{depth}_mean_rank_displacement",
        f"stage_{depth}_hpool_full_score_correlation",
    ]
    values = [
        float(hpool_docs[0] == cascade_docs[0]),
        len(set(hpool_docs[:10]) & set(cascade_docs[:10])) / 10.0,
        float(
            np.mean(
                [
                    abs(position - cascade_positions[doc_id])
                    for position, doc_id in enumerate(hpool_docs)
                ]
            )
            / depth
        ),
        correlation,
    ]
    for boundary in (1, 5, 10):
        names.extend(
            (
                f"stage_{depth}_full_standard_score_at_{boundary}",
                f"stage_{depth}_full_standard_margin_at_{boundary}",
            )
        )
        values.extend(
            (
                float(standard[boundary - 1]),
                float(
                    (cascade_values[boundary - 1] - cascade_values[boundary])
                    / scale
                ),
            )
        )
    return names, np.nan_to_num(np.asarray(values, dtype=np.float64))


def _fold(query_id: str, folds: int) -> int:
    value = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16)
    return value % folds


def _budget_matched_regret(
    average_rows: float,
    fixed_regret: dict[int, float],
) -> dict[str, Any]:
    if average_rows <= PLANS[0]:
        return {"lower_plan": 20, "upper_plan": 20, "expected_regret": fixed_regret[20]}
    for lower, upper in zip(PLANS[:-1], PLANS[1:], strict=True):
        if average_rows <= upper:
            upper_fraction = (average_rows - lower) / (upper - lower)
            return {
                "lower_plan": lower,
                "upper_plan": upper,
                "upper_plan_fraction": upper_fraction,
                "expected_regret": (
                    (1.0 - upper_fraction) * fixed_regret[lower]
                    + upper_fraction * fixed_regret[upper]
                ),
            }
    return {"lower_plan": 1110, "upper_plan": 1110, "expected_regret": 0.0}


def analyze(
    *,
    qrels_path: Path,
    queries_path: Path,
    full_path: Path,
    hpool_path: Path,
    agc_path: Path,
    cascade_paths: dict[int, Path],
    folds: int = 5,
    ndcg_tolerance: float = 0.001,
    risk_levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05),
    bootstrap_resamples: int = 10_000,
    seed: int = 20260805,
) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    queries = {
        str(row["query_id"]): str(row["query"])
        for row in (json.loads(line) for line in queries_path.read_text().splitlines())
    }
    full = load_scored_ranking(full_path, expected_depth=100)
    hpool = load_scored_ranking(hpool_path, expected_depth=100)
    agc = load_scored_ranking(agc_path, expected_depth=100)
    cascades = {
        depth: load_scored_ranking(path, expected_depth=depth)
        for depth, path in cascade_paths.items()
    }
    query_ids = sorted(qrels, key=lambda value: (not value.isdigit(), value))
    required = set(query_ids)
    for name, ranking in {"full": full, "hpool": hpool, "agc": agc, **{str(k): v for k, v in cascades.items()}}.items():
        if set(ranking) != required:
            raise ValueError(f"query IDs differ for {name}")

    metrics: dict[int, np.ndarray] = {}
    metrics[1110] = np.asarray(
        [_query_metrics([doc for doc, _ in full[q]], qrels[q], (10,))["ndcg_at_10"] for q in query_ids]
    )
    for depth in (20, 50, 100):
        metrics[depth] = np.asarray(
            [_query_metrics([doc for doc, _ in cascades[depth][q]], qrels[q], (10,))["ndcg_at_10"] for q in query_ids]
        )
    full_metric = metrics[1110]
    fixed_regret = {depth: float(np.mean(full_metric - values)) for depth, values in metrics.items()}

    feature_sets: dict[str, dict[int, np.ndarray]] = {}
    feature_names: dict[str, dict[str, list[str]]] = {}
    for feature_set, include_agc in (("hpool_only", False), ("hpool_plus_agc", True)):
        rows = []
        names = None
        for query_id in query_ids:
            names, row = _query_features(
                hpool[query_id],
                queries[query_id],
                agc=agc[query_id] if include_agc else None,
            )
            rows.append(row)
        matrix = np.stack(rows)
        feature_sets[feature_set] = {depth: matrix for depth in (20, 50, 100)}
        feature_names[feature_set] = {
            str(depth): names or [] for depth in (20, 50, 100)
        }
    progressive_rows: dict[int, list[np.ndarray]] = {
        depth: [] for depth in (20, 50, 100)
    }
    progressive_names: dict[str, list[str]] = {}
    for query_id in query_ids:
        base_names, base_values = _query_features(
            hpool[query_id], queries[query_id], agc=None
        )
        for depth in (20, 50, 100):
            stage_names, stage_values = _progressive_features(
                hpool[query_id], cascades[depth][query_id]
            )
            progressive_names[str(depth)] = base_names + stage_names
            progressive_rows[depth].append(
                np.concatenate((base_values, stage_values))
            )
    feature_sets["progressive_hpool_full"] = {
        depth: np.stack(rows) for depth, rows in progressive_rows.items()
    }
    feature_names["progressive_hpool_full"] = progressive_names

    teacher_unsafe = {}
    for depth in (20, 50, 100):
        teacher_unsafe[depth] = np.asarray(
            [
                not (
                    {doc for doc, _ in full[query_id][:10]}
                    <= {doc for doc, _ in hpool[query_id][:depth]}
                )
                for query_id in query_ids
            ],
            dtype=float,
        )
    relevance_unsafe = {
        depth: ((full_metric - metrics[depth]) > ndcg_tolerance).astype(float)
        for depth in (20, 50, 100)
    }
    fold_ids = np.asarray([_fold(query_id, folds) for query_id in query_ids])

    policies: dict[str, Any] = {}
    decision_rows: dict[str, list[dict[str, Any]]] = {}
    for feature_set, features_by_depth in feature_sets.items():
        for target_name, targets in (
            ("full_top10_escape", teacher_unsafe),
            ("train_qrel_ndcg_regret", relevance_unsafe),
        ):
            for risk in risk_levels:
                policy_name = f"{feature_set}__{target_name}__risk_{risk:g}"
                selected = np.full(len(query_ids), 1110, dtype=int)
                predicted_risk = np.ones((len(query_ids), 3), dtype=np.float64)
                thresholds_by_fold: dict[str, dict[str, float]] = {}
                for fold in range(folds):
                    train = fold_ids != fold
                    test = fold_ids == fold
                    test_positions = np.where(test)[0]
                    thresholds_by_fold[str(fold)] = {}
                    probabilities = {}
                    for offset, depth in enumerate((20, 50, 100)):
                        features = features_by_depth[depth]
                        model = fit_logistic(features[train], targets[depth][train])
                        train_scores = predict_logistic(model, features[train])
                        threshold = admission_threshold(
                            train_scores,
                            targets[depth][train],
                            maximum_empirical_risk=risk,
                        )
                        thresholds_by_fold[str(fold)][str(depth)] = threshold
                        probabilities[depth] = predict_logistic(model, features[test])
                        predicted_risk[test, offset] = probabilities[depth]
                    for local_position, query_position in enumerate(test_positions):
                        for depth in (20, 50, 100):
                            if probabilities[depth][local_position] <= thresholds_by_fold[str(fold)][str(depth)]:
                                selected[query_position] = depth
                                break
                candidate_metric = np.asarray(
                    [metrics[int(plan)][position] for position, plan in enumerate(selected)]
                )
                summary = regret_summary(
                    full_metric,
                    candidate_metric,
                    catastrophic_threshold=0.10,
                    seed=seed,
                    resamples=bootstrap_resamples,
                )
                average_rows = float(selected.mean())
                matched_mixture = _budget_matched_regret(
                    average_rows, fixed_regret
                )
                lower_plan = int(matched_mixture["lower_plan"])
                upper_plan = int(matched_mixture["upper_plan"])
                upper_fraction = float(
                    matched_mixture.get("upper_plan_fraction", 0.0)
                )
                mixture_metric = (
                    (1.0 - upper_fraction) * metrics[lower_plan]
                    + upper_fraction * metrics[upper_plan]
                )
                adaptive_improvement = (
                    (full_metric - mixture_metric)
                    - (full_metric - candidate_metric)
                )
                summary.update(
                    {
                        "average_cold_rows": average_rows,
                        "average_cold_bytes": average_rows * 11_529_200,
                        "plan_counts": {
                            str(plan): int(np.sum(selected == plan)) for plan in PLANS
                        },
                        "harm_over_tolerance_queries": int(
                            np.sum((full_metric - candidate_metric) > ndcg_tolerance)
                        ),
                        "budget_matched_query_independent_mixture": matched_mixture,
                        "adaptive_improvement_over_mixture": bootstrap_mean_interval(
                            adaptive_improvement,
                            seed=seed + 1,
                            resamples=bootstrap_resamples,
                        ),
                        "thresholds_by_fold": thresholds_by_fold,
                        "uses_qrels_for_policy_training": target_name
                        == "train_qrel_ndcg_regret",
                        "held_out_fold_qrels_used_for_policy": False,
                        "agc_locator_cost_charged": False,
                        "deployable_as_measured": feature_set
                        in ("hpool_only", "progressive_hpool_full"),
                        "decision_timing": (
                            "before_cold_access"
                            if feature_set != "progressive_hpool_full"
                            else "after_materializing_current_depth_before_escalation"
                        ),
                    }
                )
                policies[policy_name] = summary
                keep_decisions = (
                    feature_set == "progressive_hpool_full"
                    and target_name == "full_top10_escape"
                    and risk == 0.05
                ) or (
                    feature_set == "hpool_only"
                    and target_name == "train_qrel_ndcg_regret"
                    and risk == 0.0
                )
                if keep_decisions:
                    decision_rows[policy_name] = [
                        {
                            "query_id": query_id,
                            "fold": int(fold_ids[position]),
                            "plan": int(selected[position]),
                            "full_ndcg_at_10": float(full_metric[position]),
                            "candidate_ndcg_at_10": float(candidate_metric[position]),
                            "regret": float(
                                full_metric[position] - candidate_metric[position]
                            ),
                            "teacher_escape_at_20": bool(
                                teacher_unsafe[20][position]
                            ),
                            "teacher_escape_at_50": bool(
                                teacher_unsafe[50][position]
                            ),
                            "teacher_escape_at_100": bool(
                                teacher_unsafe[100][position]
                            ),
                            "predicted_risk_at_20": float(
                                predicted_risk[position, 0]
                            ),
                            "predicted_risk_at_50": float(
                                predicted_risk[position, 1]
                            ),
                            "predicted_risk_at_100": float(
                                predicted_risk[position, 2]
                            ),
                        }
                        for position, query_id in enumerate(query_ids)
                    ]

    qrel_oracle_plan = np.full(len(query_ids), 1110, dtype=int)
    teacher_oracle_plan = np.full(len(query_ids), 1110, dtype=int)
    for position in range(len(query_ids)):
        for depth in (20, 50, 100):
            if max(full_metric[position] - metrics[depth][position], 0.0) <= ndcg_tolerance:
                qrel_oracle_plan[position] = depth
                break
        for depth in (20, 50, 100):
            if not teacher_unsafe[depth][position]:
                teacher_oracle_plan[position] = depth
                break

    return {
        "schema_version": 1,
        "protocol": "crossfit-query-adaptive-omni-cascade-2026-08-05",
        "queries": len(query_ids),
        "folds": folds,
        "fold_assignment": "sha256(query_id)_prefix_mod_folds",
        "ndcg_regret_tolerance": ndcg_tolerance,
        "regret_sign": "positive_means_plan_worse_than_full",
        "plans": {
            "cold_rows": {str(plan): plan for plan in PLANS},
            "bytes_per_full_document_row": 11_529_200,
            "hot_hpool_bytes_constant_and_excluded": 581_990_515,
        },
        "fixed": {
            str(depth): regret_summary(
                full_metric,
                values,
                catastrophic_threshold=0.10,
                seed=seed,
                resamples=bootstrap_resamples,
            )
            for depth, values in metrics.items()
        },
        "oracle": {
            "qrel_quality_oracle": {
                "average_cold_rows": float(qrel_oracle_plan.mean()),
                "plan_counts": {str(plan): int(np.sum(qrel_oracle_plan == plan)) for plan in PLANS},
                "uses_test_qrels": True,
                "upper_bound_only": True,
            },
            "full_top10_teacher_oracle": {
                "average_cold_rows": float(teacher_oracle_plan.mean()),
                "plan_counts": {str(plan): int(np.sum(teacher_oracle_plan == plan)) for plan in PLANS},
                "uses_qrels": False,
                "uses_full_ranking_teacher": True,
            },
        },
        "feature_names": feature_names,
        "policies": policies,
        "per_query_operating_points": decision_rows,
        "artifacts": {
            "qrels": {"path": str(qrels_path), "sha256": _sha256(qrels_path)},
            "queries": {"path": str(queries_path), "sha256": _sha256(queries_path)},
            "full": {"path": str(full_path), "sha256": _sha256(full_path)},
            "hpool": {"path": str(hpool_path), "sha256": _sha256(hpool_path)},
            "agc": {"path": str(agc_path), "sha256": _sha256(agc_path)},
            **{
                f"cascade_{depth}": {"path": str(path), "sha256": _sha256(path)}
                for depth, path in cascade_paths.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--full-ranking", type=Path, required=True)
    parser.add_argument("--hpool-ranking", type=Path, required=True)
    parser.add_argument("--agc-ranking", type=Path, required=True)
    parser.add_argument("--cascade20-ranking", type=Path, required=True)
    parser.add_argument("--cascade50-ranking", type=Path, required=True)
    parser.add_argument("--cascade100-ranking", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ndcg-tolerance", type=float, default=0.001)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        qrels_path=args.qrels,
        queries_path=args.queries,
        full_path=args.full_ranking,
        hpool_path=args.hpool_ranking,
        agc_path=args.agc_ranking,
        cascade_paths={
            20: args.cascade20_ranking,
            50: args.cascade50_ranking,
            100: args.cascade100_ranking,
        },
        folds=args.folds,
        ndcg_tolerance=args.ndcg_tolerance,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "oracle": report["oracle"],
                "policies": {
                    name: {
                        key: value
                        for key, value in row.items()
                        if key
                        in (
                            "average_cold_rows",
                            "mean_regret",
                            "ci95_lower",
                            "ci95_upper",
                            "harm_over_tolerance_queries",
                            "plan_counts",
                            "budget_matched_query_independent_mixture",
                        )
                    }
                    for name, row in report["policies"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
