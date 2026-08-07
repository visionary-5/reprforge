"""Cheap-feature coverage audit for text-locator candidate escapes."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from reprforge.policy_replay import ReplayData


VISUAL_TYPES = frozenset({"chart", "figure", "image", "table"})
NUMERIC_FEATURES = (
    "native_text_chars",
    "ocr_text_chars",
    "has_image",
    "page_id",
    "grayscale_entropy",
    "grayscale_std",
    "edge_energy",
    "nonwhite_fraction",
    "image_aspect_log",
    "image_height",
    "image_width",
)


def repair_events(
    data: ReplayData, query_ids: Sequence[str], *, depth: int
) -> list[dict[str, Any]]:
    query_map = {query.query_id: query for query in data.queries}
    events = []
    for query_id in query_ids:
        query = query_map[str(query_id)]
        candidates = query.candidate_item_ids or tuple(item.item_id for item in data.items)
        text = sorted(
            candidates,
            key=lambda item_id: (-data.scores["text"][query.query_id][item_id], item_id),
        )
        visual = sorted(
            candidates,
            key=lambda item_id: (-data.scores["image"][query.query_id][item_id], item_id),
        )
        if query.relevant_item_ids & set(text[:depth]):
            continue
        repair = query.relevant_item_ids & set(visual[:depth])
        for item_id in sorted(repair):
            events.append({"query_id": query.query_id, "item_id": item_id})
    return events


def item_feature_rows(
    item_rows: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str],
    *,
    content_types: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    by_id = {str(row["item_id"]): row for row in item_rows}
    if content_types is None:
        content_types = sorted(
            {
                str(by_id[item_id].get("content_type", "unknown")).lower()
                for item_id in item_ids
            }
        )
    else:
        content_types = list(content_types)
    names = list(NUMERIC_FEATURES) + [f"content_type={value}" for value in content_types]
    matrix = []
    for item_id in item_ids:
        row = by_id[item_id]
        construction = row.get("construction_features") or {}
        numeric = {
            "native_text_chars": len(str(row.get("native_text") or "")),
            "ocr_text_chars": len(str(row.get("ocr_text") or "")),
            "has_image": float(bool(row.get("has_image"))),
            "page_id": float(row.get("page_id") or 0),
            **{name: float(construction.get(name) or 0.0) for name in NUMERIC_FEATURES[4:]},
        }
        content = str(row.get("content_type", "unknown")).lower()
        matrix.append(
            [float(numeric[name]) for name in NUMERIC_FEATURES]
            + [float(content == value) for value in content_types]
        )
    return np.asarray(matrix, dtype=np.float64), names


def fit_ridge_risk(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized_train = (train_x - mean) / scale
    normalized_test = (test_x - mean) / scale
    train_design = np.column_stack((np.ones(len(train_x)), normalized_train))
    test_design = np.column_stack((np.ones(len(test_x)), normalized_test))
    positives = max(int(np.sum(train_y > 0)), 1)
    negatives = max(int(np.sum(train_y <= 0)), 1)
    weights = np.where(train_y > 0, negatives / positives, 1.0)
    gram = train_design.T @ (weights[:, None] * train_design)
    regularizer = np.eye(gram.shape[0]) * float(ridge_lambda)
    regularizer[0, 0] = 0.0
    target = train_design.T @ (weights * train_y)
    coefficients = np.linalg.solve(gram + regularizer, target)
    return test_design @ coefficients


def select_by_policy(
    *,
    policy: str,
    item_ids: Sequence[str],
    item_rows: Sequence[Mapping[str, Any]],
    item_documents: Mapping[str, int],
    count: int,
    ridge_scores: np.ndarray,
    repair_items: set[str],
    seed: int,
) -> set[str]:
    count = min(max(int(count), 0), len(item_ids))
    if count == 0:
        return set()
    by_id = {str(row["item_id"]): row for row in item_rows}
    if policy == "random":
        selected = np.random.default_rng(seed).choice(len(item_ids), count, replace=False)
        return {item_ids[int(index)] for index in selected}
    if policy == "document_uniform":
        groups: dict[int, list[str]] = defaultdict(list)
        for item_id in item_ids:
            groups[int(item_documents[item_id])].append(item_id)
        selected: list[str] = []
        position = 0
        documents = sorted(groups)
        while len(selected) < count:
            progressed = False
            for document in documents:
                values = groups[document]
                if position < len(values):
                    selected.append(values[position])
                    progressed = True
                    if len(selected) >= count:
                        break
            if not progressed:
                break
            position += 1
        return set(selected)
    if policy == "visual_type":
        score = np.asarray(
            [float(str(by_id[item_id].get("content_type", "")).lower() in VISUAL_TYPES) for item_id in item_ids]
        )
    elif policy == "text_scarcity":
        score = np.asarray(
            [
                -math.log1p(
                    len(str(by_id[item_id].get("native_text") or ""))
                    + len(str(by_id[item_id].get("ocr_text") or ""))
                )
                for item_id in item_ids
            ]
        )
    elif policy == "ridge_risk":
        score = ridge_scores
    elif policy == "repair_oracle":
        score = np.asarray([float(item_id in repair_items) for item_id in item_ids])
    else:
        raise ValueError(f"unsupported policy: {policy}")
    order = np.lexsort((np.asarray(item_ids), -score))
    return {item_ids[int(index)] for index in order[:count]}


def evaluate_cover(
    selected: set[str],
    events: Sequence[Mapping[str, str]],
    *,
    all_item_ids: Sequence[str],
    item_rows: Sequence[Mapping[str, Any]],
    item_domains: Mapping[str, str],
) -> dict[str, Any]:
    repair_items = {str(event["item_id"]) for event in events}
    repair_queries = {str(event["query_id"]) for event in events}
    covered_events = [event for event in events if str(event["item_id"]) in selected]
    covered_queries = {str(event["query_id"]) for event in covered_events}
    by_id = {str(row["item_id"]): row for row in item_rows}
    full_cost = sum(float(by_id[item_id]["route_costs"]["image"]["encode_ms"]) for item_id in all_item_ids)
    selected_cost = sum(float(by_id[item_id]["route_costs"]["image"]["encode_ms"]) for item_id in selected)
    domain_rows = {}
    for domain in sorted(set(item_domains.values())):
        domain_events = [event for event in events if item_domains[str(event["item_id"])] == domain]
        domain_queries = {str(event["query_id"]) for event in domain_events}
        domain_covered = [event for event in domain_events if str(event["item_id"]) in selected]
        domain_covered_queries = {str(event["query_id"]) for event in domain_covered}
        if domain_events:
            domain_rows[domain] = {
                "repair_events": len(domain_events),
                "repair_queries": len(domain_queries),
                "event_recall": len(domain_covered) / len(domain_events),
                "query_coverage": len(domain_covered_queries) / len(domain_queries),
            }
    return {
        "selected_items": len(selected),
        "selected_fraction": len(selected) / len(all_item_ids),
        "selected_encode_cost_fraction": selected_cost / full_cost if full_cost else None,
        "repair_items": len(repair_items),
        "repair_item_recall": len(selected & repair_items) / len(repair_items) if repair_items else None,
        "repair_event_recall": len(covered_events) / len(events) if events else None,
        "repair_query_coverage": len(covered_queries) / len(repair_queries) if repair_queries else None,
        "selected_precision": len(selected & repair_items) / len(selected) if selected else None,
        "domains": domain_rows,
    }
