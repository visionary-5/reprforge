#!/usr/bin/env python3
"""Evaluate value-aware visual materialization on an exported score surface."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.bm25 import scores as bm25_scores
from reprforge.partial_vlm_materialization import (
    ScoreSurface,
    evaluate_selection,
    evaluate_text_only,
    evaluate_visual_only,
)
from reprforge.value_aware_materialization import (
    CompilerConfig,
    compile_value_aware_index,
    evaluate_compiled_index,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_exported_surface(
    score_root: Path, dataset_root: Path
) -> tuple[ScoreSurface, list[dict[str, Any]], list[dict[str, Any]]]:
    query_ids = [str(value) for value in json.loads(
        (score_root / "query_ids.json").read_text()
    )]
    doc_ids: list[str] = []
    score_blocks = []
    for doc_id_path in sorted(score_root.glob("doc_ids-*.json")):
        suffix = doc_id_path.stem.split("-")[-1]
        score_path = score_root / f"scores-{suffix}.npy"
        ids = [str(value) for value in json.loads(doc_id_path.read_text())]
        scores = np.load(score_path)
        if scores.shape != (len(query_ids), len(ids)):
            raise ValueError(f"score/ID mismatch in shard {suffix}")
        doc_ids.extend(ids)
        score_blocks.append(np.asarray(scores, dtype=np.float32))
    if not score_blocks or len(doc_ids) != len(set(doc_ids)):
        raise ValueError("score export is empty or has duplicate pages")
    visual_scores = np.concatenate(score_blocks, axis=1)

    corpus_rows = read_jsonl(dataset_root / "corpus.jsonl")
    query_rows = read_jsonl(dataset_root / "queries.jsonl")
    qrel_rows = read_jsonl(dataset_root / "qrels.jsonl")
    corpus_by_id = {str(row["docid"]): row for row in corpus_rows}
    query_by_id = {str(row["query_id"]): row for row in query_rows}
    if set(doc_ids) != set(corpus_by_id) or set(query_ids) != set(query_by_id):
        raise ValueError("dataset IDs differ from exported score IDs")
    documents = [str(corpus_by_id[doc_id].get("text", "")) for doc_id in doc_ids]
    queries = [str(query_by_id[query_id]["query"]) for query_id in query_ids]
    text_scores, posting_bytes, _ = bm25_scores(documents, queries)
    qrels = np.zeros_like(visual_scores, dtype=np.float32)
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    doc_positions = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    for row in qrel_rows:
        qrels[
            query_positions[str(row["query_id"])],
            doc_positions[str(row["doc_id"])],
        ] = float(row["relevance"])
    surface = ScoreSurface(
        name=dataset_root.name,
        query_ids=np.asarray(query_ids),
        corpus_ids=np.asarray(doc_ids),
        text_scores=text_scores,
        visual_scores=visual_scores,
        qrels=qrels,
        text_bytes=posting_bytes.astype(np.float64),
        visual_bytes=np.ones(len(doc_ids), dtype=np.float64),
        visual_encode_ms=np.ones(len(doc_ids), dtype=np.float64),
        input_sha256={},
    )
    return surface, corpus_rows, query_rows


def load_features(path: Path, surface: ScoreSurface) -> np.ndarray:
    rows = read_jsonl(path)
    by_id = {str(row["doc_id"]): row for row in rows}
    if set(by_id) != set(map(str, surface.corpus_ids)):
        raise ValueError("feature IDs differ from score surface")
    return np.asarray(
        [
            [
                math.log1p(float(by_id[str(doc_id)]["text_chars"])),
                float(by_id[str(doc_id)]["grayscale_entropy"]),
                float(by_id[str(doc_id)]["edge_energy"]),
                float(by_id[str(doc_id)]["locator_disagreement"]),
                math.log1p(float(by_id[str(doc_id)]["history_candidate_count"])),
            ]
            for doc_id in surface.corpus_ids
        ],
        dtype=np.float64,
    )


def load_split(path: Path, surface: ScoreSurface) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    assignments = payload["queries"]
    evaluation_fold = int(payload["evaluation_fold"])
    history, evaluation = [], []
    for position, query_id in enumerate(surface.query_ids):
        target = evaluation if int(assignments[str(query_id)]) == evaluation_fold else history
        target.append(position)
    if not history or not evaluation:
        raise ValueError("history/evaluation split is empty")
    return np.asarray(history, dtype=np.int32), np.asarray(evaluation, dtype=np.int32)


def static_order(features: np.ndarray, strategy: str, seed: int) -> np.ndarray:
    pages = np.arange(len(features), dtype=np.int32)
    if strategy == "random":
        return np.random.default_rng(seed).permutation(pages)
    if strategy == "text_scarcity":
        return pages[np.lexsort((pages, features[:, 0]))]
    if strategy == "visual_complexity":
        value = features[:, 1] + features[:, 2]
        return pages[np.lexsort((pages, -value))]
    if strategy == "locator_disagreement":
        return pages[np.lexsort((pages, -features[:, 3]))]
    if strategy == "history_frequency":
        return pages[np.lexsort((pages, -features[:, 4]))]
    raise ValueError(f"unsupported static strategy: {strategy}")


def tune_calibration(
    surface: ScoreSurface,
    history: np.ndarray,
    anchors: np.ndarray,
) -> tuple[CompilerConfig, list[dict[str, float]]]:
    rows = []
    for quantile in (0.5, 0.75, 0.9, 0.95):
        for weight in (0.25, 0.5, 1.0, 2.0):
            config = CompilerConfig(
                anchor_pages=len(anchors),
                calibration_quantile=quantile,
                visual_weight=weight,
            )
            quality = evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=config,
            )["mean_ndcg_at_10"]
            rows.append(
                {
                    "calibration_quantile": quantile,
                    "visual_weight": weight,
                    "history_full_ndcg_at_10": quality,
                }
            )
    rows.sort(
        key=lambda row: (
            -row["history_full_ndcg_at_10"],
            row["calibration_quantile"],
            row["visual_weight"],
        )
    )
    best = rows[0]
    return CompilerConfig(
        anchor_pages=len(anchors),
        calibration_quantile=float(best["calibration_quantile"]),
        visual_weight=float(best["visual_weight"]),
    ), rows


def tune_support_calibration(
    surface: ScoreSurface,
    history: np.ndarray,
    anchors: np.ndarray,
) -> tuple[CompilerConfig, list[dict[str, float]]]:
    rows = []
    for alpha in (0.01, 0.05, 0.1, 0.2):
        for weight in (1.0, 2.0, 4.0, 8.0):
            config = CompilerConfig(
                anchor_pages=len(anchors),
                calibration_quantile=0.9,
                visual_weight=weight,
                two_way_centering=True,
                familywise_alpha=alpha,
            )
            quality = evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=config,
                calibration_queries=history,
            )["mean_ndcg_at_10"]
            rows.append(
                {
                    "familywise_alpha": alpha,
                    "visual_weight": weight,
                    "history_full_ndcg_at_10": quality,
                }
            )
    rows.sort(
        key=lambda row: (
            -row["history_full_ndcg_at_10"],
            row["familywise_alpha"],
            row["visual_weight"],
        )
    )
    best = rows[0]
    return CompilerConfig(
        anchor_pages=len(anchors),
        calibration_quantile=0.9,
        visual_weight=float(best["visual_weight"]),
        two_way_centering=True,
        familywise_alpha=float(best["familywise_alpha"]),
    ), rows


def tune_anchor_rank(
    surface: ScoreSurface,
    history: np.ndarray,
    anchors: np.ndarray,
) -> tuple[CompilerConfig, list[dict[str, float]]]:
    rows = []
    for smoothing in (0.1, 0.5, 1.0):
        for weight in (0.5, 1.0, 2.0):
            config = CompilerConfig(
                anchor_pages=len(anchors),
                fusion="anchor_rank",
                visual_weight=weight,
                anchor_rank_smoothing=smoothing,
            )
            quality = evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=config,
            )["mean_ndcg_at_10"]
            rows.append(
                {
                    "anchor_rank_smoothing": smoothing,
                    "visual_weight": weight,
                    "history_full_ndcg_at_10": quality,
                }
            )
    rows.sort(
        key=lambda row: (
            -row["history_full_ndcg_at_10"],
            row["anchor_rank_smoothing"],
            row["visual_weight"],
        )
    )
    best = rows[0]
    return CompilerConfig(
        anchor_pages=len(anchors),
        fusion="anchor_rank",
        visual_weight=float(best["visual_weight"]),
        anchor_rank_smoothing=float(best["anchor_rank_smoothing"]),
    ), rows


def run_matrix(
    surface: ScoreSurface,
    features: np.ndarray,
    history: np.ndarray,
    evaluation: np.ndarray,
    *,
    budgets: list[float],
    anchor_pages: int,
    seed: int,
) -> dict[str, Any]:
    anchor_order = np.random.default_rng(seed).permutation(surface.pages)
    anchors = np.sort(anchor_order[:anchor_pages])
    tuned, tuning_rows = tune_calibration(surface, history, anchors)
    support_tuned, support_tuning_rows = tune_support_calibration(
        surface, history, anchors
    )
    rank_tuned, rank_tuning_rows = tune_anchor_rank(surface, history, anchors)
    baselines = {
        "text_only": {
            "history": evaluate_text_only(surface, history),
            "evaluation": evaluate_text_only(surface, evaluation),
        },
        "visual_only": {
            "history": evaluate_visual_only(surface, history),
            "evaluation": evaluate_visual_only(surface, evaluation),
        },
        "full_calibrated_hybrid": {
            "history": evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=tuned,
            ),
            "evaluation": evaluate_compiled_index(
                surface,
                evaluation,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=tuned,
                calibration_queries=history,
            ),
        },
        "full_two_way_support_calibrated": {
            "history": evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=support_tuned,
                calibration_queries=history,
            ),
            "evaluation": evaluate_compiled_index(
                surface,
                evaluation,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=support_tuned,
                calibration_queries=history,
            ),
        },
        "full_anchor_rank_estimated": {
            "history": evaluate_compiled_index(
                surface,
                history,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=rank_tuned,
            ),
            "evaluation": evaluate_compiled_index(
                surface,
                evaluation,
                admitted_pages=np.arange(surface.pages),
                anchor_pages=anchors,
                config=rank_tuned,
            ),
        },
    }
    curves: dict[str, dict[str, Any]] = {
        name: {} for name in (
            "random",
            "text_scarcity",
            "visual_complexity",
            "locator_disagreement",
            "history_frequency",
            "value_aware_beta0",
            "value_aware_beta1",
            "naive_local_zscore_random",
            "random_two_way_support",
            "text_scarcity_two_way_support",
            "value_aware_support_beta0",
            "value_aware_support_beta1",
            "random_anchor_rank",
            "text_scarcity_anchor_rank",
            "history_verified_anchor_rank",
            "value_aware_anchor_rank_beta0",
            "value_aware_anchor_rank_beta1",
        )
    }
    history_relevance = np.sum(surface.qrels[history] > 0, axis=0).astype(np.float64)
    pages = np.arange(surface.pages, dtype=np.int32)
    history_verified_order = pages[
        np.lexsort((pages, -history_relevance))
    ]
    feedback_features = np.column_stack((features, np.log1p(history_relevance)))
    for fraction in budgets:
        count = max(anchor_pages, int(math.ceil(fraction * surface.pages)))
        budget_key = str(fraction)
        for strategy in (
            "random",
            "text_scarcity",
            "visual_complexity",
            "locator_disagreement",
            "history_frequency",
        ):
            order = static_order(features, strategy, seed)
            selected = list(map(int, anchors))
            selected_set = set(selected)
            for page in order:
                selected_set.add(int(page))
                if len(selected_set) >= count:
                    break
            selected_array = np.asarray(sorted(selected_set), dtype=np.int32)
            curves[strategy][budget_key] = {
                "materialized_pages": len(selected_array),
                "history": evaluate_compiled_index(
                    surface,
                    history,
                    admitted_pages=selected_array,
                    anchor_pages=anchors,
                    config=tuned,
                ),
                "evaluation": evaluate_compiled_index(
                    surface,
                    evaluation,
                    admitted_pages=selected_array,
                    anchor_pages=anchors,
                    config=tuned,
                    calibration_queries=history,
                ),
            }
            if strategy in ("random", "text_scarcity"):
                support_name = f"{strategy}_two_way_support"
                curves[support_name][budget_key] = {
                    "materialized_pages": len(selected_array),
                    "history": evaluate_compiled_index(
                        surface,
                        history,
                        admitted_pages=selected_array,
                        anchor_pages=anchors,
                        config=support_tuned,
                        calibration_queries=history,
                    ),
                    "evaluation": evaluate_compiled_index(
                        surface,
                        evaluation,
                        admitted_pages=selected_array,
                        anchor_pages=anchors,
                        config=support_tuned,
                        calibration_queries=history,
                    ),
                }
                rank_name = f"{strategy}_anchor_rank"
                curves[rank_name][budget_key] = {
                    "materialized_pages": len(selected_array),
                    "history": evaluate_compiled_index(
                        surface,
                        history,
                        admitted_pages=selected_array,
                        anchor_pages=anchors,
                        config=rank_tuned,
                    ),
                    "evaluation": evaluate_compiled_index(
                        surface,
                        evaluation,
                        admitted_pages=selected_array,
                        anchor_pages=anchors,
                        config=rank_tuned,
                    ),
                }
        verified = set(map(int, anchors))
        for page in history_verified_order:
            verified.add(int(page))
            if len(verified) >= count:
                break
        verified_array = np.asarray(sorted(verified), dtype=np.int32)
        curves["history_verified_anchor_rank"][budget_key] = {
            "materialized_pages": len(verified_array),
            "history_feedback_semantics": "perfect historical verifier upper bound",
            "history": evaluate_compiled_index(
                surface,
                history,
                admitted_pages=verified_array,
                anchor_pages=anchors,
                config=rank_tuned,
            ),
            "evaluation": evaluate_compiled_index(
                surface,
                evaluation,
                admitted_pages=verified_array,
                anchor_pages=anchors,
                config=rank_tuned,
            ),
        }
        random_selected = static_order(features, "random", seed)[:count]
        curves["naive_local_zscore_random"][budget_key] = {
            "materialized_pages": count,
            "history": evaluate_selection(
                surface, history, random_selected, fusion="zscore"
            ),
            "evaluation": evaluate_selection(
                surface, evaluation, random_selected, fusion="zscore"
            ),
        }
        for beta, name in ((0.0, "value_aware_beta0"), (1.0, "value_aware_beta1")):
            config = CompilerConfig(
                anchor_pages=anchor_pages,
                calibration_quantile=tuned.calibration_quantile,
                visual_weight=tuned.visual_weight,
                exploration=beta,
                seed=seed,
            )
            compiled = compile_value_aware_index(
                surface,
                history,
                page_features=features,
                page_costs=np.ones(surface.pages),
                maximum_cost=float(count),
                config=config,
            )
            curves[name][budget_key] = {
                "materialized_pages": len(compiled["probed_pages"]),
                "admitted_pages": len(compiled["admitted_pages"]),
                "rejected_pages": len(compiled["rejected_pages"]),
                "history": {
                    "mean_ndcg_at_10": compiled["history_mean_ndcg_at_10"]
                },
                "evaluation": evaluate_compiled_index(
                    surface,
                    evaluation,
                    admitted_pages=compiled["admitted_pages"],
                    anchor_pages=compiled["anchor_pages"],
                    config=config,
                    calibration_queries=history,
                ),
                "trace": compiled["trace"],
            }
        for beta, name in (
            (0.0, "value_aware_support_beta0"),
            (1.0, "value_aware_support_beta1"),
        ):
            config = CompilerConfig(
                anchor_pages=anchor_pages,
                calibration_quantile=support_tuned.calibration_quantile,
                visual_weight=support_tuned.visual_weight,
                exploration=beta,
                two_way_centering=True,
                familywise_alpha=support_tuned.familywise_alpha,
                seed=seed,
            )
            compiled = compile_value_aware_index(
                surface,
                history,
                page_features=features,
                page_costs=np.ones(surface.pages),
                maximum_cost=float(count),
                config=config,
            )
            curves[name][budget_key] = {
                "materialized_pages": len(compiled["probed_pages"]),
                "admitted_pages": len(compiled["admitted_pages"]),
                "rejected_pages": len(compiled["rejected_pages"]),
                "history": {
                    "mean_ndcg_at_10": compiled["history_mean_ndcg_at_10"]
                },
                "evaluation": evaluate_compiled_index(
                    surface,
                    evaluation,
                    admitted_pages=compiled["admitted_pages"],
                    anchor_pages=compiled["anchor_pages"],
                    config=config,
                    calibration_queries=history,
                ),
                "trace": compiled["trace"],
            }
        for beta, name in (
            (0.0, "value_aware_anchor_rank_beta0"),
            (1.0, "value_aware_anchor_rank_beta1"),
        ):
            config = CompilerConfig(
                anchor_pages=anchor_pages,
                fusion="anchor_rank",
                visual_weight=rank_tuned.visual_weight,
                anchor_rank_smoothing=rank_tuned.anchor_rank_smoothing,
                exploration=beta,
                seed=seed,
            )
            compiled = compile_value_aware_index(
                surface,
                history,
                page_features=feedback_features,
                page_costs=np.ones(surface.pages),
                maximum_cost=float(count),
                config=config,
            )
            curves[name][budget_key] = {
                "materialized_pages": len(compiled["probed_pages"]),
                "admitted_pages": len(compiled["admitted_pages"]),
                "rejected_pages": len(compiled["rejected_pages"]),
                "history_feedback_semantics": "qrels stand in for a perfect historical verifier",
                "history": {
                    "mean_ndcg_at_10": compiled["history_mean_ndcg_at_10"]
                },
                "evaluation": evaluate_compiled_index(
                    surface,
                    evaluation,
                    admitted_pages=compiled["admitted_pages"],
                    anchor_pages=compiled["anchor_pages"],
                    config=config,
                ),
                "trace": compiled["trace"],
            }
    return {
        "domain": surface.name,
        "queries": surface.queries,
        "pages": surface.pages,
        "history_queries": len(history),
        "evaluation_queries": len(evaluation),
        "anchor_pages": anchor_pages,
        "seed": seed,
        "tuned_calibration": {
            "quantile": tuned.calibration_quantile,
            "visual_weight": tuned.visual_weight,
            "selection_rule": "maximum Full calibrated hybrid nDCG on history only",
            "grid": tuning_rows,
        },
        "tuned_support_calibration": {
            "familywise_alpha": support_tuned.familywise_alpha,
            "visual_weight": support_tuned.visual_weight,
            "two_way_centering": True,
            "selection_rule": "maximum Full support-calibrated nDCG on history only",
            "grid": support_tuning_rows,
        },
        "tuned_anchor_rank": {
            "smoothing": rank_tuned.anchor_rank_smoothing,
            "visual_weight": rank_tuned.visual_weight,
            "selection_rule": "maximum Full estimated-rank RRF nDCG on history only",
            "grid": rank_tuning_rows,
        },
        "baselines": baselines,
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--query-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.02, 0.05, 0.1, 0.2))
    parser.add_argument("--anchor-pages", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    surface, _, _ = load_exported_surface(args.score_root, args.dataset_root)
    features = load_features(args.features, surface)
    history, evaluation = load_split(args.query_splits, surface)
    result = run_matrix(
        surface,
        features,
        history,
        evaluation,
        budgets=list(args.budgets),
        anchor_pages=args.anchor_pages,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "domain": result["domain"],
        "output": str(args.output),
        "tuned_calibration": result["tuned_calibration"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
