"""Deterministic page organizations and parent-level retrieval aggregation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


BBox = tuple[int, int, int, int]


def fixed_quadrants(
    width: int, height: int, *, overlap_fraction: float
) -> list[BBox]:
    """Return four deterministic overlapping quadrant boxes."""

    if width < 2 or height < 2:
        return [(0, 0, width, height)]
    overlap_x = int(round(width * overlap_fraction / 2))
    overlap_y = int(round(height * overlap_fraction / 2))
    middle_x = width // 2
    middle_y = height // 2
    return [
        (0, 0, min(width, middle_x + overlap_x), min(height, middle_y + overlap_y)),
        (max(0, middle_x - overlap_x), 0, width, min(height, middle_y + overlap_y)),
        (0, max(0, middle_y - overlap_y), min(width, middle_x + overlap_x), height),
        (max(0, middle_x - overlap_x), max(0, middle_y - overlap_y), width, height),
    ]


def _content_bbox(ink: np.ndarray) -> BBox:
    rows = np.flatnonzero(np.any(ink, axis=1))
    columns = np.flatnonzero(np.any(ink, axis=0))
    if not len(rows) or not len(columns):
        return (0, 0, ink.shape[1], ink.shape[0])
    return (int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1))


def _best_whitespace_cut(
    ink: np.ndarray,
    box: BBox,
    *,
    minimum_region_fraction: float,
    minimum_gap_fraction: float,
) -> tuple[str, int, float] | None:
    left, top, right, bottom = box
    region = ink[top:bottom, left:right]
    height, width = region.shape
    choices: list[tuple[str, int, float]] = []
    for axis, density, extent, offset in (
        ("horizontal", region.mean(axis=1), height, top),
        ("vertical", region.mean(axis=0), width, left),
    ):
        minimum_side = max(1, int(round(extent * minimum_region_fraction)))
        allowed = np.arange(minimum_side, extent - minimum_side + 1)
        if not len(allowed):
            continue
        whitespace_mask = density[allowed] <= 0.1
        padded = np.pad(whitespace_mask.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        minimum_gap = max(1, int(round(extent * minimum_gap_fraction)))
        runs = [(start, stop) for start, stop in zip(starts, stops) if stop - start >= minimum_gap]
        if not runs:
            continue
        start, stop = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
        position = int(allowed[(start + stop - 1) // 2])
        score = float(np.mean(1.0 - density[allowed[start:stop]]))
        choices.append((axis, offset + position, score))
    return max(choices, key=lambda item: item[2]) if choices else None


def xycut_regions(
    image: Image.Image,
    *,
    maximum_units: int,
    analysis_maximum_side: int,
    ink_threshold: int,
    minimum_region_fraction: float,
    minimum_whitespace_gap_fraction: float,
    crop_padding_fraction: float,
) -> list[BBox]:
    """Split a page recursively at strong whitespace valleys.

    The proposer is intentionally deterministic and model-free. It is a physical
    organization control, not a claimed document-layout parser.
    """

    gray = image.convert("L")
    scale = min(1.0, analysis_maximum_side / max(gray.size))
    if scale < 1.0:
        gray = gray.resize(
            (max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
            Image.Resampling.BILINEAR,
        )
    ink = np.asarray(gray) < ink_threshold
    boxes = [_content_bbox(ink)]
    while len(boxes) < maximum_units:
        options = []
        for index, box in enumerate(boxes):
            cut = _best_whitespace_cut(
                ink,
                box,
                minimum_region_fraction=minimum_region_fraction,
                minimum_gap_fraction=minimum_whitespace_gap_fraction,
            )
            if cut is not None:
                options.append((cut[2], index, cut))
        if not options:
            break
        _, index, (axis, position, _) = max(options)
        left, top, right, bottom = boxes.pop(index)
        if axis == "horizontal":
            boxes.extend([(left, top, right, position), (left, position, right, bottom)])
        else:
            boxes.extend([(left, top, position, bottom), (position, top, right, bottom)])
    original_width, original_height = image.size
    padding = crop_padding_fraction
    result = []
    for left, top, right, bottom in boxes:
        left = left / scale
        right = right / scale
        top = top / scale
        bottom = bottom / scale
        pad_x = (right - left) * padding
        pad_y = (bottom - top) * padding
        result.append(
            (
                max(0, int(left - pad_x)),
                max(0, int(top - pad_y)),
                min(original_width, int(np.ceil(right + pad_x))),
                min(original_height, int(np.ceil(bottom + pad_y))),
            )
        )
    return sorted(result, key=lambda box: (box[1], box[0], box[3], box[2]))


def deterministic_neutral_order(
    doc_ids: Iterable[str], *, protocol_id: str, domain: str
) -> list[str]:
    return sorted(
        map(str, doc_ids),
        key=lambda doc_id: hashlib.sha256(
            f"{protocol_id}\0{domain}\0{doc_id}".encode()
        ).digest(),
    )


def aggregate_unit_ranking(
    ranked_units: Sequence[tuple[str, float]],
    unit_to_parent: Mapping[str, str],
) -> list[tuple[str, float]]:
    """Aggregate unit scores to unique parents with max score and stable ties."""

    parent_scores: dict[str, float] = {}
    for unit_id, score in ranked_units:
        parent = unit_to_parent[unit_id]
        parent_scores[parent] = max(score, parent_scores.get(parent, float("-inf")))
    return sorted(parent_scores.items(), key=lambda item: (-item[1], item[0]))


def dcg(relevances: Sequence[float]) -> float:
    values = np.asarray(relevances, dtype=np.float64)
    if not len(values):
        return 0.0
    return float(np.sum((2.0**values - 1.0) / np.log2(np.arange(2, len(values) + 2))))


def parent_metrics(
    ranking: Sequence[str], qrels: Mapping[str, float], *, depth: int
) -> dict[str, Any]:
    observed = [float(qrels.get(doc_id, 0.0)) for doc_id in ranking[:depth]]
    ideal = sorted(map(float, qrels.values()), reverse=True)[:depth]
    denominator = dcg(ideal)
    return {
        "ndcg": dcg(observed) / denominator if denominator else 0.0,
        "hit": bool(any(value > 0 for value in observed)),
    }
