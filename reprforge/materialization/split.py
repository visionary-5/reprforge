"""Frozen fit/calibration/test split handling without relevance access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WorkloadSplit:
    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]
    query_ids: tuple[str, ...]
    assignment: str
    seed: int

    def validate(self) -> None:
        groups = [set(self.fit), set(self.calibration), set(self.test)]
        if any(not group for group in groups):
            raise ValueError("fit, calibration, and test must all be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("split partitions overlap")
        expected = set(range(len(self.query_ids)))
        if set.union(*groups) != expected:
            raise ValueError("split partitions do not cover every query exactly once")

    def test_orders(self, seeds: Sequence[int]) -> dict[str, np.ndarray]:
        self.validate()
        base = np.asarray(self.test, dtype=np.int32)
        output = {"dataset_order": base.copy()}
        for seed in seeds:
            output[f"random_seed_{int(seed)}"] = np.random.default_rng(int(seed)).permutation(base)
        return output


def load_frozen_split(
    path: Path,
    query_ids: Sequence[str],
    *,
    fit_folds: Sequence[int] = (1, 2, 3),
    calibration_folds: Sequence[int] = (4,),
) -> WorkloadSplit:
    payload = json.loads(path.read_text())
    assignments = payload["queries"]
    evaluation_fold = int(payload["evaluation_fold"])
    fit_fold_set = set(map(int, fit_folds))
    calibration_fold_set = set(map(int, calibration_folds))
    if evaluation_fold in fit_fold_set | calibration_fold_set:
        raise ValueError("evaluation fold must not be used for fit or calibration")
    if fit_fold_set & calibration_fold_set:
        raise ValueError("fit and calibration folds overlap")
    fit: list[int] = []
    calibration: list[int] = []
    test: list[int] = []
    for position, query_id in enumerate(map(str, query_ids)):
        if query_id not in assignments:
            raise ValueError(f"query missing from frozen assignment: {query_id}")
        fold = int(assignments[query_id])
        if fold == evaluation_fold:
            test.append(position)
        elif fold in calibration_fold_set:
            calibration.append(position)
        elif fold in fit_fold_set:
            fit.append(position)
        else:
            raise ValueError(f"unassigned fold in frozen split: {fold}")
    split = WorkloadSplit(
        fit=tuple(fit),
        calibration=tuple(calibration),
        test=tuple(test),
        query_ids=tuple(map(str, query_ids)),
        assignment=str(payload.get("assignment", "unknown")),
        seed=int(payload["seed"]),
    )
    split.validate()
    return split
