#!/usr/bin/env python3
"""Cross-fit an exact content-type representation compiler on MMDocIR."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_cohort_certificates import _mmdocir_cube
from reprforge.type_policy_compiler import crossfit_type_policy_compiler


def _query_folds(query_ids: tuple[str, ...], count: int = 5) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % count
            for value in query_ids
        ],
        dtype=np.int16,
    )


def _document_folds(groups: list[int], count: int = 5) -> np.ndarray:
    frequencies = Counter(groups)
    loads = [0] * count
    assignment = {}
    for group, size in sorted(frequencies.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(count), key=lambda value: (loads[value], value))
        assignment[group] = fold
        loads[fold] += size
    return np.asarray([assignment[group] for group in groups], dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdocir-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cube, _, metadata = _mmdocir_cube(args.mmdocir_root)
    item_rows = metadata["item_rows"]
    route_costs = {
        route: np.asarray(
            [row["route_costs"][route]["index_bytes"] for row in item_rows],
            dtype=np.float64,
        )
        for route in cube.routes
    }
    common = {
        "item_types": [str(row["content_type"]) for row in item_rows],
        "candidate_indices": metadata["candidate_indices"],
        "route_costs": route_costs,
    }
    report = {
        "schema_version": 1,
        "query_recurrence_crossfit": crossfit_type_policy_compiler(
            cube, fold_ids=_query_folds(cube.query_ids), **common
        ),
        "source_document_crossfit": crossfit_type_policy_compiler(
            cube,
            fold_ids=_document_folds(metadata["query_document_indices"]),
            **common,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
