#!/usr/bin/env python3
"""Physically execute reusable pair probes on the IRPAPERS page mirror.

The frozen score surface supplies only the locator candidates and the teacher
ranking.  Every visual score visible to a policy is recomputed from a page
representation constructed in the same charged run.  A probe is therefore a
real physical-design action: its page embedding remains resident and may
answer every later edge that references the page.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.boundary_admission import execute_boundary_plan
from reprforge.cohort_compiler import CohortCompiler
from reprforge.irpapers_benchmark import IRPapersColPaliBackend
from reprforge.pairwise_view_admission import (
    select_independent_pages,
)
from reprforge.reusable_pair_probe import (
    FrozenPairScoreProvider,
    PairScoreProvider,
    select_reusable_pair_probes,
)
from tools.analyze_pairwise_view_admission import _balanced_group_folds
from tools.analyze_reusable_pair_probe import (
    _agreement,
    _boundary_pairs,
    _recall,
)
from tools.analyze_sparse_risk_admission import _load_cost_model
from tools.run_pairwise_admission_physical import _candidate_surface


POLICIES = (
    "active_pair_15",
    "static_pair_15",
    "static_pair_atomic_15",
    "independent_20",
)


class LazyImageCorpus(Sequence[bytes]):
    """Read page bytes only when an admitted view is physically constructed."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = tuple(paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [path.read_bytes() for path in self._paths[index]]
        return self._paths[index].read_bytes()


def _load_page_paths(
    manifest: Path,
    corpus_ids: Sequence[str],
    required_pages: set[int],
) -> tuple[Path, ...]:
    root = manifest.parent
    by_id: dict[str, Path] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            item_id = f"{row['pdf_id']}_{row['page_number']}"
            if item_id in by_id:
                raise ValueError(f"duplicate page {item_id!r} at line {line_number}")
            path = root / str(row["image_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            by_id[item_id] = path
    corpus_set = set(corpus_ids)
    extra = set(by_id) - corpus_set
    required_ids = {corpus_ids[index] for index in required_pages}
    missing = required_ids - set(by_id)
    if missing or extra:
        raise ValueError(
            f"page mirror differs from score surface: "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    # The compiler indexes corpus positions, but only candidate pages can be
    # requested.  Non-candidate placeholders are never read.
    placeholder = manifest
    return tuple(by_id.get(item_id, placeholder) for item_id in corpus_ids)


@dataclass
class PhysicalCounters:
    query_encode_ms: float = 0.0
    visual_encode_ms: float = 0.0
    visual_score_ms: float = 0.0
    visual_pages_encoded: int = 0
    visual_encoder_calls: int = 0
    visual_score_pairs: int = 0
    resident_vector_bytes: int = 0


class PhysicalPairScoreProvider(PairScoreProvider):
    """Leakage-safe provider backed by real ColPali construction and MaxSim."""

    def __init__(
        self,
        *,
        candidates: np.ndarray,
        query_texts: Sequence[str],
        corpus_ids: Sequence[str],
        corpus_images: Sequence[bytes],
        backend: IRPapersColPaliBackend,
    ) -> None:
        self._candidates = np.asarray(candidates, dtype=np.int64)
        if self._candidates.ndim != 2:
            raise ValueError("candidates must be two-dimensional")
        self._corpus_ids = tuple(corpus_ids)
        self._backend = backend
        self._compiler = CohortCompiler(
            corpus_ids=self._corpus_ids,
            corpus_texts=("",) * len(self._corpus_ids),
            corpus_images=corpus_images,
            backend=backend,
            candidate_k=self._candidates.shape[1],
            top_k=self._candidates.shape[1],
            cache_policy="resident",
        )
        began = time.perf_counter()
        encoded = backend.encode_queries(query_texts)
        self._query_embeddings = encoded.embeddings
        self.counters = PhysicalCounters(
            query_encode_ms=(time.perf_counter() - began) * 1000.0
        )
        self._scores: dict[tuple[int, int], float] = {}
        self.materialization_calls = 0

    @property
    def materialized_pages(self) -> frozenset[int]:
        return frozenset(
            index
            for index, item_id in enumerate(self._corpus_ids)
            if item_id in self._compiler.resident_item_ids
        )

    def materialize(self, pages: Sequence[int] | set[int]) -> None:
        requested = sorted({int(page) for page in pages})
        invalid = [page for page in requested if not 0 <= page < len(self._corpus_ids)]
        if invalid:
            raise ValueError(f"invalid page indices: {invalid[:5]}")
        resident_before = self.materialized_pages
        new_pages = [page for page in requested if page not in resident_before]
        if not new_pages:
            return
        metrics = self._compiler.materialize_items(
            self._corpus_ids[page] for page in new_pages
        )
        self.counters.visual_encode_ms += float(metrics["visual_encode_ms"])
        self.counters.visual_pages_encoded += int(metrics["visual_pages_encoded"])
        self.counters.visual_encoder_calls += int(metrics["visual_encoder_calls"])
        self.counters.resident_vector_bytes += int(metrics["resident_vector_bytes"])
        documents = self._compiler.resident_embeddings(
            [self._corpus_ids[page] for page in new_pages]
        )
        score_began = time.perf_counter()
        scores = self._backend.score(self._query_embeddings, documents)
        self.counters.visual_score_ms += (time.perf_counter() - score_began) * 1000.0
        if len(scores) != len(self._query_embeddings):
            raise RuntimeError("visual backend returned incomplete query scores")
        for query_index, row in enumerate(scores):
            if len(row) != len(new_pages):
                raise RuntimeError("visual backend returned incomplete page scores")
            for page, value in zip(new_pages, row, strict=True):
                self._scores[(query_index, page)] = float(value)
        self.counters.visual_score_pairs += len(self._query_embeddings) * len(new_pages)
        self.materialization_calls += 1

    def score(self, query_index: int, candidate_rank: int) -> float:
        page = int(self._candidates[query_index, candidate_rank])
        try:
            return self._scores[(int(query_index), page)]
        except KeyError as error:
            raise RuntimeError("attempted to read an unmaterialized visual score") from error

    def candidate_score_matrix(self) -> np.ndarray:
        values = np.zeros(self._candidates.shape, dtype=np.float64)
        residents = self.materialized_pages
        for query in range(len(self._candidates)):
            for rank, page_value in enumerate(self._candidates[query]):
                page = int(page_value)
                if page in residents:
                    values[query, rank] = self._scores[(query, page)]
        return values


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_policy(
    *,
    policy: str,
    fold: int,
    repetition: int,
    test: np.ndarray,
    candidates: np.ndarray,
    locator: np.ndarray,
    teacher: np.ndarray,
    query_texts: Sequence[str],
    gold_ids: Sequence[str],
    corpus_ids: list[str],
    corpus_images: Sequence[bytes],
    backend: IRPapersColPaliBackend,
    cost_model,
    cutoff: int,
    round_page_limit: int,
    warmup_page_limit: int,
    minimum_observations: int,
) -> dict[str, Any]:
    episode_candidates = candidates[test]
    episode_locator = locator[test]
    episode_teacher = teacher[test]
    episode_queries = [query_texts[int(index)] for index in test]
    episode_gold = [gold_ids[int(index)] for index in test]
    eligible = len(set(int(page) for page in episode_candidates.flat))
    budget_fraction = 0.20 if policy == "independent_20" else 0.15
    page_budget = math.floor(budget_fraction * eligible)
    provider = PhysicalPairScoreProvider(
        candidates=episode_candidates,
        query_texts=episode_queries,
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        backend=backend,
    )
    wall_began = time.perf_counter()
    plan_metrics: dict[str, Any] = {}
    if policy == "independent_20":
        pairs = _boundary_pairs(episode_candidates, episode_locator, cutoff=cutoff)
        selected = select_independent_pages(
            pairs,
            page_budget=page_budget,
        ).selected_pages
        provider.materialize(selected)
    elif policy == "static_pair_atomic_15":
        # Static risk never consumes observed scores.  Plan against a
        # leakage-checking zero surface, then compile the final set in one
        # physical submission instead of replaying artificial planning rounds.
        planning_provider = FrozenPairScoreProvider(
            episode_candidates,
            np.zeros(episode_candidates.shape, dtype=np.float64),
        )
        time_budget = cost_model.estimate(
            pages=page_budget,
            score_events=0,
        ).total_ms
        plan = select_reusable_pair_probes(
            episode_candidates,
            episode_locator,
            planning_provider,
            cost_model,
            cutoff=cutoff,
            time_budget_ms=time_budget,
            empirical_updates=False,
            round_page_limit=round_page_limit,
            warmup_page_limit=warmup_page_limit,
            minimum_observations=minimum_observations,
        )
        selected = plan.selected_pages
        provider.materialize(selected)
        plan_metrics = {
            "observed_pairs": plan.observed_pair_count,
            "selector_iterations": plan.iterations,
            "selector_materialization_rounds": plan.materialization_rounds,
            "physical_compilation_rounds": 1,
        }
    else:
        time_budget = cost_model.estimate(
            pages=page_budget,
            score_events=0,
        ).total_ms
        plan = select_reusable_pair_probes(
            episode_candidates,
            episode_locator,
            provider,
            cost_model,
            cutoff=cutoff,
            time_budget_ms=time_budget,
            empirical_updates=policy == "active_pair_15",
            round_page_limit=round_page_limit,
            warmup_page_limit=warmup_page_limit,
            minimum_observations=minimum_observations,
        )
        selected = plan.selected_pages
        plan_metrics = {
            "observed_pairs": plan.observed_pair_count,
            "selector_iterations": plan.iterations,
            "selector_materialization_rounds": plan.materialization_rounds,
        }
    construction_and_probe_ms = (time.perf_counter() - wall_began) * 1000.0
    visual = provider.candidate_score_matrix()
    rankings, work = execute_boundary_plan(
        episode_candidates,
        episode_locator,
        visual,
        selected_pages=set(selected),
        visual_prior_by_rank=np.zeros(episode_candidates.shape[1]),
        cutoff=cutoff,
    )
    counters = provider.counters
    return {
        "repetition": repetition,
        "held_out_fold": fold,
        "policy": policy,
        "queries": len(test),
        "eligible_pages": eligible,
        "budget_fraction": budget_fraction,
        "page_budget": page_budget,
        "selected_pages": len(selected),
        "quality": {
            "recall_5": _recall(rankings, corpus_ids, episode_gold),
            "exact_teacher_agreement": _agreement(rankings, episode_teacher),
        },
        "physical": {
            "construction_and_probe_ms": construction_and_probe_ms,
            "query_encode_ms": counters.query_encode_ms,
            "charged_total_ms": construction_and_probe_ms
            + counters.query_encode_ms,
            "visual_encode_ms": counters.visual_encode_ms,
            "visual_score_ms": counters.visual_score_ms,
            "visual_pages_encoded": counters.visual_pages_encoded,
            "visual_encoder_calls": counters.visual_encoder_calls,
            "materialization_calls": provider.materialization_calls,
            "visual_score_pairs": counters.visual_score_pairs,
            "resident_vector_bytes": counters.resident_vector_bytes,
            "final_visual_candidate_events": int(work["visual_candidate_events"]),
        },
        **plan_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--physical-runs", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--round-page-limit", type=int, default=8)
    parser.add_argument("--warmup-page-limit", type=int, default=2)
    parser.add_argument("--minimum-observations", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.candidate_k <= args.cutoff:
        raise ValueError("candidate_k must exceed cutoff")

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    candidates, locator, _, _, teacher = _candidate_surface(
        corpus_ids,
        np.asarray(surface["bm25_scores"], dtype=np.float64),
        np.asarray(surface["visual_scores"], dtype=np.float64),
        candidate_k=args.candidate_k,
        cutoff=args.cutoff,
    )
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    if len(query_rows) != len(candidates):
        raise ValueError("query metadata and score surface differ in length")
    folds = _balanced_group_folds(
        np.asarray([str(row["pdf_id"]) for row in query_rows])
    )
    query_texts = [str(row["question"]) for row in query_rows]
    gold_ids = [str(row["dataset_id"]) for row in query_rows]
    required_pages = {int(page) for page in candidates.flat}
    page_paths = _load_page_paths(
        args.page_manifest,
        corpus_ids,
        required_pages,
    )
    corpus_images = LazyImageCorpus(page_paths)
    cost_model, cost_diagnostics = _load_cost_model(
        args.physical_runs,
        batch_size=args.batch_size,
    )

    model_began = time.perf_counter()
    backend = IRPapersColPaliBackend(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
    )
    model_load_seconds = time.perf_counter() - model_began

    import torch

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "physical-reusable-pair-probe",
        "configuration": {
            "policies": list(POLICIES),
            "active_static_budget_fraction": 0.15,
            "independent_budget_fraction": 0.20,
            "candidate_k": args.candidate_k,
            "cutoff": args.cutoff,
            "round_page_limit": args.round_page_limit,
            "warmup_page_limit": args.warmup_page_limit,
            "minimum_observations": args.minimum_observations,
            "minimum_observations_selected_on_irpapers": True,
            "repetitions": args.repetitions,
            "schedule": "fold-interleaved-alternating-policy-order",
            "selection_uses_qrels": False,
            "unmaterialized_visual_scores_visible": False,
        },
        "cost_model": {
            "batch_size": cost_model.batch_size,
            "setup_ms": cost_model.setup_ms,
            "page_ms": cost_model.page_ms,
            "batch_ms": cost_model.batch_ms,
            "score_event_ms": cost_model.score_event_ms,
            **cost_diagnostics,
        },
        "resource_contract": {
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "visible_gpu_count": torch.cuda.device_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "model_load_seconds": model_load_seconds,
            "page_mirror": str(args.page_manifest.resolve()),
            "score_surface": str(args.score_surface.resolve()),
        },
        "runs": [],
    }
    fold_values = sorted(set(int(value) for value in folds))
    for repetition in range(args.repetitions):
        for fold in fold_values:
            order = list(POLICIES)
            shift = (repetition + fold) % len(order)
            order = order[shift:] + order[:shift]
            test = np.flatnonzero(folds == fold)
            for policy in order:
                payload["runs"].append(
                    _run_policy(
                        policy=policy,
                        fold=fold,
                        repetition=repetition,
                        test=test,
                        candidates=candidates,
                        locator=locator,
                        teacher=teacher,
                        query_texts=query_texts,
                        gold_ids=gold_ids,
                        corpus_ids=corpus_ids,
                        corpus_images=corpus_images,
                        backend=backend,
                        cost_model=cost_model,
                        cutoff=args.cutoff,
                        round_page_limit=args.round_page_limit,
                        warmup_page_limit=args.warmup_page_limit,
                        minimum_observations=args.minimum_observations,
                    )
                )
                _atomic_write(args.output, payload)


if __name__ == "__main__":
    main()
