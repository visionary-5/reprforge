#!/usr/bin/env python3
"""Evaluate a query-conditioned, set-level visual refinement policy.

The policy deliberately chooses a cohort size rather than scoring page
upgrades independently. It predicts a quality curve from semantically similar
training queries, then selects the point that maximizes predicted quality
minus a visual-work price. Test qrels and future query order are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.intervention_utility import (
    _ndcg_row,
    build_intervention_events,
)
from reprforge.progressive_oracle import (
    FrozenTrace,
    load_trace,
    rank_order,
    validate_pair,
)


DEFAULT_COHORT_SIZES = (0, 5, 10, 20, 50, 100, 200, 1110)
NEIGHBOR_GRID = (5, 10, 20, 40, 80)
PRICE_GRID = tuple(float(value) for value in np.linspace(0.0, 0.30, 61))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_query_texts(path: Path, query_ids: Sequence[str]) -> list[str]:
    """Load public query strings while keeping them out of result artifacts."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional benchmark extra
        raise RuntimeError("query metadata requires the ViDoRe/pyarrow extra") from error
    table = pq.read_table(path, columns=["query_id", "query"])
    lookup = {
        str(query_id): str(query)
        for query_id, query in zip(
            table.column("query_id").to_pylist(),
            table.column("query").to_pylist(),
        )
    }
    missing = [str(value) for value in query_ids if str(value) not in lookup]
    if missing:
        raise ValueError(f"query metadata is missing {len(missing)} trace identifiers")
    return [lookup[str(value)] for value in query_ids]


def query_folds(query_ids: Sequence[str], *, fold_count: int = 5) -> np.ndarray:
    if fold_count < 3:
        raise ValueError("at least three folds are required")
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(str(value).encode("utf-8")).digest()[:4],
                "big",
            )
            % fold_count
            for value in query_ids
        ],
        dtype=np.int16,
    )


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [f"{left}__{right}" for left, right in zip(words, words[1:])]


def tfidf_profiles(
    query_texts: Sequence[str],
    train_queries: np.ndarray,
) -> np.ndarray:
    """Build a label-free unigram/bigram profile with train-only vocabulary."""

    documents = [_tokens(value) for value in query_texts]
    document_frequency: Counter[str] = Counter()
    for query in train_queries:
        document_frequency.update(set(documents[int(query)]))
    kept = sorted(
        word for word, frequency in document_frequency.items() if frequency >= 2
    )
    vocabulary = {word: index for index, word in enumerate(kept)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for query, tokens in enumerate(documents):
        for word, count in Counter(tokens).items():
            if word not in vocabulary:
                continue
            inverse_frequency = math.log(
                (1.0 + len(train_queries))
                / (1.0 + document_frequency[word])
            )
            matrix[query, vocabulary[word]] = (
                1.0 + math.log(count)
            ) * inverse_frequency
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def build_cohort_quality_curves(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    cohort_sizes: Sequence[int],
    cutoff: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    qrels = validate_pair(text, visual)
    order = rank_order(text.scores, text.corpus_ids)
    sizes = np.asarray(
        sorted({min(text.scores.shape[1], max(0, int(value))) for value in cohort_sizes}),
        dtype=np.int32,
    )
    quality = np.zeros((text.scores.shape[0], len(sizes)), dtype=np.float64)
    for query in range(text.scores.shape[0]):
        for column, size in enumerate(sizes):
            scores = text.scores[query].copy()
            pages = order[query, :size]
            scores[pages] = visual.scores[query, pages]
            quality[query, column] = _ndcg_row(
                scores,
                qrels[query],
                text.corpus_ids,
                cutoff=cutoff,
            )
    return sizes, quality


def predict_quality_curves(
    profiles: np.ndarray,
    quality: np.ndarray,
    train_queries: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    prediction = np.zeros_like(quality)
    count = min(neighbors, len(train_queries))
    for query in range(len(profiles)):
        similarity = profiles[train_queries] @ profiles[query]
        nearest = np.argsort(-similarity, kind="stable")[:count]
        weights = np.maximum(similarity[nearest], 0.0) + 1e-6
        prediction[query] = np.average(
            quality[train_queries[nearest]],
            axis=0,
            weights=weights,
        )
    return prediction


def choose_cohorts(
    predicted_quality: np.ndarray,
    cohort_sizes: np.ndarray,
    *,
    visual_work_price: float,
) -> np.ndarray:
    objective = predicted_quality - visual_work_price * (
        cohort_sizes[None, :] / max(int(cohort_sizes[-1]), 1)
    )
    return np.argmax(objective, axis=1).astype(np.int16)


def _cache_replay(
    cohorts: Sequence[Sequence[int]],
    visual: FrozenTrace,
    *,
    capacity: int,
    two_hit: bool,
) -> dict[str, Any]:
    cache: OrderedDict[int, None] = OrderedDict()
    touches: Counter[int] = Counter()
    encodes = 0
    hits = 0
    estimated_encode_ms = 0.0
    for cohort in cohorts:
        for value in cohort:
            page = int(value)
            touches[page] += 1
            if page in cache:
                hits += 1
                cache.move_to_end(page)
                continue
            encodes += 1
            estimated_encode_ms += float(visual.encode_ms[page])
            if not two_hit or touches[page] >= 2:
                cache[page] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    events = encodes + hits
    return {
        "capacity_pages": capacity,
        "two_hit_admission": two_hit,
        "candidate_events": events,
        "visual_encodes": encodes,
        "cache_hits": hits,
        "cache_hit_fraction": hits / events if events else 0.0,
        "final_resident_pages": len(cache),
        "estimated_batched_equivalent_encode_ms": estimated_encode_ms,
    }


def _fold_result(
    fold: int,
    folds: np.ndarray,
    profiles: np.ndarray,
    quality: np.ndarray,
    cohort_sizes: np.ndarray,
    text_order: np.ndarray,
    visual: FrozenTrace,
    *,
    page_budget: int,
) -> dict[str, Any]:
    test_queries = np.flatnonzero(folds == fold)
    validation_queries = np.flatnonzero(folds == ((fold + 1) % int(folds.max() + 1)))
    train_queries = np.flatnonzero(
        (folds != fold) & (folds != ((fold + 1) % int(folds.max() + 1)))
    )

    # Refit query vocabulary for every fold. The caller's profile is ignored
    # only when it was built globally; analyze_cohort_policy passes a per-fold
    # matrix through this parameter.
    best: tuple[float, float, int, float, np.ndarray] | None = None
    for neighbors in NEIGHBOR_GRID:
        predicted = predict_quality_curves(
            profiles,
            quality,
            train_queries,
            neighbors=neighbors,
        )
        for price in PRICE_GRID:
            choice = choose_cohorts(
                predicted,
                cohort_sizes,
                visual_work_price=price,
            )
            average_pages = float(cohort_sizes[choice[validation_queries]].mean())
            if average_pages > page_budget:
                continue
            validation_quality = float(
                quality[validation_queries, choice[validation_queries]].mean()
            )
            candidate = (
                validation_quality,
                -average_pages,
                neighbors,
                price,
                choice,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise RuntimeError("no cohort policy satisfies the validation budget")
    validation_quality, negative_validation_pages, neighbors, price, choice = best

    allowed_fixed = np.flatnonzero(cohort_sizes <= page_budget)
    fixed_column = int(
        allowed_fixed[
            np.argmax(quality[validation_queries][:, allowed_fixed].mean(axis=0))
        ]
    )
    selected_sizes = cohort_sizes[choice[test_queries]]
    cohorts = [
        text_order[query, : int(cohort_sizes[choice[query]])].tolist()
        for query in test_queries
    ]
    flattened = [page for cohort in cohorts for page in cohort]
    return {
        "fold": fold,
        "train_queries": int(len(train_queries)),
        "validation_queries": int(len(validation_queries)),
        "test_queries": int(len(test_queries)),
        "selected_neighbors": neighbors,
        "selected_visual_work_price": price,
        "validation_ndcg@10": validation_quality,
        "validation_average_cohort_pages": -negative_validation_pages,
        "test": {
            "text_ndcg@10": float(quality[test_queries, 0].mean()),
            "full_visual_ndcg@10": float(quality[test_queries, -1].mean()),
            "best_fixed_cohort_pages": int(cohort_sizes[fixed_column]),
            "best_fixed_ndcg@10": float(quality[test_queries, fixed_column].mean()),
            "policy_ndcg@10": float(
                quality[test_queries, choice[test_queries]].mean()
            ),
            "per_query_oracle_ndcg@10": float(
                quality[test_queries].max(axis=1).mean()
            ),
            "average_cohort_pages": float(selected_sizes.mean()),
            "cohort_size_histogram": {
                str(int(size)): int(np.count_nonzero(selected_sizes == size))
                for size in cohort_sizes
            },
            "candidate_events": len(flattened),
            "unique_visual_pages": len(set(flattened)),
            "unbounded_reuse_fraction": (
                1.0 - len(set(flattened)) / len(flattened)
                if flattened
                else 0.0
            ),
            "lru": _cache_replay(
                cohorts,
                visual,
                capacity=page_budget,
                two_hit=False,
            ),
            "two_hit_lru": _cache_replay(
                cohorts,
                visual,
                capacity=page_budget,
                two_hit=True,
            ),
        },
    }


def _nonadditivity(
    text: FrozenTrace,
    visual: FrozenTrace,
    quality: np.ndarray,
    cohort_sizes: np.ndarray,
    *,
    candidate_k: int,
) -> dict[str, Any]:
    column = int(np.flatnonzero(cohort_sizes == candidate_k)[0])
    events = build_intervention_events(
        text,
        visual,
        candidate_k=candidate_k,
        cutoff=10,
    )
    additive = np.zeros(text.scores.shape[0], dtype=np.float64)
    np.add.at(additive, events.query_position, events.utility)
    actual = quality[:, column] - quality[:, 0]
    interaction = actual - additive
    return {
        "candidate_k": candidate_k,
        "actual_group_delta_mean": float(actual.mean()),
        "sum_individual_delta_mean": float(additive.mean()),
        "interaction_mean": float(interaction.mean()),
        "interaction_absolute_mean": float(np.abs(interaction).mean()),
        "queries_abs_interaction_gt_0_01_fraction": float(
            np.mean(np.abs(interaction) > 0.01)
        ),
        "group_and_additive_sign_disagree_fraction": float(
            np.mean(np.sign(actual) != np.sign(additive))
        ),
        "actual_additive_pearson": float(np.corrcoef(actual, additive)[0, 1]),
    }


def analyze_cohort_policy(
    text: FrozenTrace,
    visual: FrozenTrace,
    query_texts: Sequence[str],
    *,
    cohort_sizes: Sequence[int] = DEFAULT_COHORT_SIZES,
    page_budget: int = 333,
    fold_count: int = 5,
) -> dict[str, Any]:
    if len(query_texts) != len(text.query_ids):
        raise ValueError("query text and trace lengths differ")
    sizes, quality = build_cohort_quality_curves(
        text,
        visual,
        cohort_sizes=cohort_sizes,
    )
    for required in (0, 20, 50, 100, text.scores.shape[1]):
        if required not in sizes:
            raise ValueError(f"cohort sizes must include {required}")
    folds = query_folds(text.query_ids.tolist(), fold_count=fold_count)
    text_order = rank_order(text.scores, text.corpus_ids)
    fold_results = []
    for fold in range(fold_count):
        validation_fold = (fold + 1) % fold_count
        train_queries = np.flatnonzero(
            (folds != fold) & (folds != validation_fold)
        )
        profiles = tfidf_profiles(query_texts, train_queries)
        fold_results.append(
            _fold_result(
                fold,
                folds,
                profiles,
                quality,
                sizes,
                text_order,
                visual,
                page_budget=page_budget,
            )
        )

    total_queries = sum(value["test_queries"] for value in fold_results)
    aggregate: dict[str, float] = {}
    fields = (
        "text_ndcg@10",
        "full_visual_ndcg@10",
        "best_fixed_ndcg@10",
        "policy_ndcg@10",
        "per_query_oracle_ndcg@10",
        "average_cohort_pages",
    )
    for field in fields:
        aggregate[field] = float(
            sum(
                value["test"][field] * value["test_queries"]
                for value in fold_results
            )
            / total_queries
        )
    aggregate["candidate_events"] = int(
        sum(value["test"]["candidate_events"] for value in fold_results)
    )
    aggregate["mean_unique_visual_pages_per_fold"] = float(
        np.mean(
            [value["test"]["unique_visual_pages"] for value in fold_results]
        )
    )
    aggregate["visual_gain_retained"] = (
        (aggregate["policy_ndcg@10"] - aggregate["text_ndcg@10"])
        / (aggregate["full_visual_ndcg@10"] - aggregate["text_ndcg@10"])
    )

    return {
        "schema_version": 1,
        "verdict": (
            "set-level policy prototype; persistence and end-to-end advantage "
            "are not established"
        ),
        "contract": {
            "folds": f"{fold_count}-fold query-id hash; next fold validates",
            "query_features": "train-vocabulary TF-IDF unigrams and bigrams",
            "policy": "nearest historical query quality curve minus visual-work price",
            "qrels": "train/validation targets only; never test features",
            "page_budget": page_budget,
            "cohort_sizes": sizes.tolist(),
            "cache_order": "official deterministic order within each held-out fold",
            "temporal_claim": False,
        },
        "workload": {
            "queries": int(text.scores.shape[0]),
            "corpus": int(text.scores.shape[1]),
        },
        "nonadditivity": [
            _nonadditivity(
                text,
                visual,
                quality,
                sizes,
                candidate_k=value,
            )
            for value in (20, 50, 100)
        ],
        "folds": fold_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-budget", type=int, default=333)
    args = parser.parse_args()

    text = load_trace(args.text_trace)
    visual = load_trace(args.visual_trace)
    query_texts = load_query_texts(args.query_metadata, text.query_ids.tolist())
    result = analyze_cohort_policy(
        text,
        visual,
        query_texts,
        page_budget=args.page_budget,
    )
    result["source"] = {
        "query_metadata_sha256": _sha256(args.query_metadata),
        "text_runtime_sha256": text.manifest["runtime_sha256"],
        "visual_runtime_sha256": visual.manifest["runtime_sha256"],
        "oracle_labels_sha256": text.manifest["oracle_labels_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
