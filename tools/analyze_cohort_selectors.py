#!/usr/bin/env python3
"""Run query-holdout cheap-surface selector probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analyze_cohort_certificates import _irpapers_cube, _mmdocir_cube
from analyze_heterogeneity_atlas import _npz_cube
from reprforge.cohort_selector import analyze_selector_probe


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(value or ""))


def _hash_bin(value: str, bins: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % bins


def _vidore_semantic_candidate_features(
    cube, dataset_root: Path
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    query_path = next((dataset_root / "queries").glob("*.parquet"))
    corpus_paths = sorted((dataset_root / "corpus").glob("*.parquet"))
    query_rows = pq.read_table(
        query_path,
        columns=[
            "query_id",
            "query",
            "query_types",
            "query_format",
            "content_type",
            "source_type",
            "query_type_for_generation",
        ],
    ).to_pylist()
    corpus_rows = []
    for path in corpus_paths:
        corpus_rows.extend(
            pq.read_table(
                path,
                columns=["corpus_id", "doc_id", "markdown", "page_number_in_doc"],
            ).to_pylist()
        )
    query_lookup = {str(row["query_id"]): row for row in query_rows}
    corpus_lookup = {str(row["corpus_id"]): row for row in corpus_rows}
    if set(cube.query_ids) - set(query_lookup) or set(cube.corpus_ids) - set(corpus_lookup):
        raise ValueError("ViDoRe semantic rows do not cover the score cube")

    document_tokens = {}
    document_frequency = {}
    for corpus_id in cube.corpus_ids:
        tokens = _tokens(corpus_lookup[corpus_id]["markdown"])
        token_set = set(tokens)
        document_tokens[corpus_id] = (tokens, token_set)
        for token in token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    corpus_count = len(cube.corpus_ids)
    positions = np.arange(corpus_count)
    output = []
    for query_index, query_id in enumerate(cube.query_ids):
        query_row = query_lookup[query_id]
        query_tokens = _tokens(query_row["query"])
        query_set = set(query_tokens)
        query_numbers = {token for token in query_set if any(c.isdigit() for c in token)}
        query_hash = np.zeros(32, dtype=np.float64)
        for token in query_tokens:
            query_hash[_hash_bin(token, 32)] += 1.0 / max(len(query_tokens), 1)
        category_hash = np.zeros(16, dtype=np.float64)
        categories = [
            *(query_row.get("query_types") or []),
            *(query_row.get("content_type") or []),
            query_row.get("query_format") or "",
            query_row.get("source_type") or "",
            query_row.get("query_type_for_generation") or "",
        ]
        for category in categories:
            category_hash[_hash_bin(str(category), 16)] = 1.0
        candidates = np.lexsort((positions, -cube.scores["text"][query_index]))[:20]
        rows = []
        for corpus_index in candidates:
            corpus_id = cube.corpus_ids[int(corpus_index)]
            document_row = corpus_lookup[corpus_id]
            markdown = document_row["markdown"] or ""
            tokens, token_set = document_tokens[corpus_id]
            overlap = query_set & token_set
            overlap_hash = np.zeros(32, dtype=np.float64)
            idf_overlap = 0.0
            for token in overlap:
                idf = np.log((corpus_count + 1.0) / (document_frequency[token] + 1.0))
                overlap_hash[_hash_bin(token, 32)] += idf
                idf_overlap += idf
            document_numbers = {
                token for token in token_set if any(c.isdigit() for c in token)
            }
            numeric = np.asarray(
                [
                    np.log1p(len(query_tokens)),
                    np.log1p(len(tokens)),
                    len(overlap) / max(len(query_set), 1),
                    len(overlap) / max(len(token_set), 1),
                    idf_overlap / max(len(query_set), 1),
                    len(query_numbers & document_numbers) / max(len(query_numbers), 1),
                    sum(char.isdigit() for char in markdown) / max(len(markdown), 1),
                    markdown.count("|") / max(len(markdown), 1),
                    markdown.count("#") / max(len(markdown), 1),
                    markdown.count("![") > 0,
                    np.log1p(max(int(document_row["page_number_in_doc"]), 0)),
                ],
                dtype=np.float64,
            )
            rows.append(np.concatenate([numeric, query_hash, overlap_hash, category_hash]))
        output.append(np.stack(rows))
    query_groups = []
    for relevance in cube.relevance:
        source_documents = sorted(
            {
                str(corpus_lookup[cube.corpus_ids[index]]["doc_id"])
                for index in relevance
            }
        )
        query_groups.append("||".join(source_documents))
    return output, tuple(query_groups)


def _balanced_group_folds(groups: tuple[str, ...], fold_count: int) -> np.ndarray:
    if fold_count < 2:
        raise ValueError("group cross-validation needs at least two folds")
    counts = Counter(groups)
    if len(counts) < fold_count:
        raise ValueError("source groups must cover every fold")
    loads = [0] * fold_count
    assignment = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        assignment[group] = fold
        loads[fold] += count
    return np.asarray([assignment[group] for group in groups], dtype=np.int16)


def _group_crossfit_probe(
    cube,
    *,
    groups: tuple[str, ...],
    pair_features: list[np.ndarray],
    pair_description: str,
    per_item_build_ms: np.ndarray,
    fold_count: int = 5,
) -> dict:
    """Evaluate selectors with source-document-disjoint outer folds."""

    fold_ids = _balanced_group_folds(groups, fold_count)
    fold_reports = []
    policies = ("top", "ridge", "random_feature")
    for fold in range(fold_count):
        roles = tuple("eval" if value == fold else "fit" for value in fold_ids)
        report = analyze_selector_probe(
            replace(cube, split_roles=roles),
            base_route="text",
            expensive_route="visual",
            candidate_k=20,
            target_k=5,
            target_metric="ndcg_at_5",
            budgets=(8,),
            per_item_build_ms=per_item_build_ms,
            candidate_pair_features=pair_features,
            pair_feature_description=pair_description,
            policies=policies,
        )
        fold_reports.append(report)
    weights = np.asarray([report["eval_queries"] for report in fold_reports], dtype=float)
    total = float(weights.sum())
    base = sum(
        weight * report["eval_base"] for weight, report in zip(weights, fold_reports)
    ) / total
    teacher = sum(
        weight * report["eval_full_teacher"]
        for weight, report in zip(weights, fold_reports)
    ) / total
    aggregate = {}
    for policy in policies:
        key = f"{policy}_b8"
        quality = sum(
            weight * report["policies"][key]["quality"]
            for weight, report in zip(weights, fold_reports)
        ) / total
        agreement = sum(
            weight * report["policies"][key]["mean_exact_position_agreement"]
            for weight, report in zip(weights, fold_reports)
        ) / total
        per_fold = [report["policies"][key]["quality"] for report in fold_reports]
        aggregate[key] = {
            "quality": float(quality),
            "full_fusion_gain_recovery": float(
                (quality - base) / (teacher - base)
                if abs(teacher - base) > 1e-12
                else 0.0
            ),
            "mean_exact_position_agreement": float(agreement),
            "quality_fold_min": float(min(per_fold)),
            "quality_fold_max": float(max(per_fold)),
            "quality_by_fold": per_fold,
        }
    return {
        "protocol": "five-fold source-document-disjoint cross-fitting",
        "fold_count": fold_count,
        "queries": int(total),
        "source_groups": len(set(groups)),
        "eval_base": float(base),
        "eval_full_teacher": float(teacher),
        "policies": aggregate,
        "folds": [
            {
                "fold": fold,
                "eval_queries": report["eval_queries"],
                "eval_source_groups": len(set(np.asarray(groups)[fold_ids == fold])),
            }
            for fold, report in enumerate(fold_reports)
        ],
        "split_uses_relevance_only_to_map_queries_to_source_documents": True,
        "selector_uses_qrels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--irpapers-surface", type=Path)
    parser.add_argument("--irpapers-queries", type=Path)
    parser.add_argument("--mmdocir-root", type=Path)
    parser.add_argument("--vidore-source-root", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "datasets": {}}
    for name in ("hr", "finance"):
        cube, _ = _npz_cube(args.data_root / name)
        visual = np.load(args.data_root / name / "visual-runtime.npz", allow_pickle=False)
        pair_features = None
        query_groups = None
        pair_description = None
        if args.vidore_source_root:
            source_name = (
                "reprforge-vidore-v3-hr"
                if name == "hr"
                else "reprforge-vidore-v3-finance-en"
            )
            pair_features, query_groups = _vidore_semantic_candidate_features(
                cube, args.vidore_source_root / source_name
            )
            pair_description = (
                "candidate query/markdown overlap, numeric and layout signals, "
                "hashed query terms, overlap terms, and declared query metadata"
            )
        result = analyze_selector_probe(
            cube,
            base_route="text",
            expensive_route="visual",
            candidate_k=20,
            target_k=5,
            target_metric="ndcg_at_5",
            per_item_build_ms=visual["encode_ms"],
            candidate_pair_features=pair_features,
            pair_feature_description=pair_description,
        )
        if pair_features is not None and query_groups is not None:
            result["source_document_crossfit"] = _group_crossfit_probe(
                cube,
                groups=query_groups,
                pair_features=pair_features,
                pair_description=pair_description,
                per_item_build_ms=visual["encode_ms"],
            )
        report["datasets"][name] = result
    if args.irpapers_surface and args.irpapers_queries:
        cube = _irpapers_cube(args.irpapers_surface, args.irpapers_queries)
        report["datasets"]["irpapers"] = analyze_selector_probe(
            cube,
            base_route="bm25",
            expensive_route="visual",
            candidate_k=20,
            target_k=5,
            target_metric="recall_at_5",
        )
    if args.mmdocir_root:
        cube, image_costs, metadata = _mmdocir_cube(args.mmdocir_root)
        result = analyze_selector_probe(
            cube,
            base_route="image-pool-25",
            expensive_route="image",
            candidate_k=10,
            target_k=5,
            target_metric="ndcg_at_5",
            budgets=(5, 8),
            per_item_build_ms=image_costs,
        )
        result["evaluation_boundary"] = "official MMDocIR within-document candidates"
        result["route_storage"] = metadata
        report["datasets"]["mmdocir_pool25_to_image"] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
