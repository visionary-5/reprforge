"""Workload-level physical plans derived from query cohort certificates."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reprforge.cohort_certificate import (
    _completed_values,
    _teacher_order_score,
    greedy_certificate,
)
from reprforge.heterogeneity_atlas import ScoreCube, paired_bootstrap_ci, query_metrics
from reprforge.landmark_probe import _completed_rerank_scores, _initial_order, _zscore


def _budgeted_documents(
    utility: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> set[int]:
    order = sorted(
        np.flatnonzero(utility > 0),
        key=lambda index: (-utility[index] / costs[index], -utility[index], int(index)),
    )
    selected = set()
    used = 0.0
    for index in order:
        cost = float(costs[index])
        if used + cost > budget:
            continue
        selected.add(int(index))
        used += cost
    return selected


def _fidelity_for_mask(
    x: np.ndarray,
    raw_expensive: np.ndarray,
    observed: np.ndarray,
    teacher_top: np.ndarray,
    target_k: int,
) -> float:
    completed = _completed_values(x, raw_expensive, observed)
    ranked = np.lexsort((np.arange(len(x)), -completed))[:target_k]
    return _teacher_order_score(ranked, teacher_top)


def _greedy_listwise_plans(
    certificates,
    base: np.ndarray,
    expensive: np.ndarray,
    fit_indices: np.ndarray,
    costs: np.ndarray,
    *,
    total_cost: float,
    budget_fractions: Sequence[float],
    candidate_k: int,
    target_k: int,
    anchors: int,
) -> dict[float, set[int]]:
    """Build one cost-normalized greedy path over qrel-free teacher fidelity."""

    anchor_positions = _initial_order(candidate_k)[:anchors]
    states = []
    occurrences: dict[int, list[tuple[int, int]]] = {}
    for local_index, query_index in enumerate(fit_indices):
        certificate = certificates[int(query_index)]
        x = _zscore(base[query_index, certificate.candidate_indices])
        raw_expensive = expensive[query_index, certificate.candidate_indices]
        observed = np.zeros(candidate_k, dtype=bool)
        observed[anchor_positions] = True
        base_fidelity = _teacher_order_score(
            np.arange(target_k), certificate.teacher_top_positions
        )
        partial_fidelity = _fidelity_for_mask(
            x,
            raw_expensive,
            observed,
            certificate.teacher_top_positions,
            target_k,
        )
        states.append(
            {
                "x": x,
                "raw_expensive": raw_expensive,
                "observed": observed,
                "teacher_top": certificate.teacher_top_positions,
                "base_fidelity": base_fidelity,
                "utility": max(base_fidelity, partial_fidelity),
            }
        )
        for position, corpus_index in enumerate(certificate.candidate_indices):
            if position in anchor_positions:
                continue
            occurrences.setdefault(int(corpus_index), []).append(
                (local_index, position)
            )

    maximum_budget = max(float(value) for value in budget_fractions) * total_cost
    selected: set[int] = set()
    path: list[tuple[int, float]] = []
    used = 0.0
    while True:
        best = None
        for corpus_index, affected in occurrences.items():
            if corpus_index in selected:
                continue
            cost = float(costs[corpus_index])
            if used + cost > maximum_budget:
                continue
            delta = 0.0
            updates = []
            for local_index, position in affected:
                state = states[local_index]
                if state["observed"][position]:
                    continue
                trial = state["observed"].copy()
                trial[position] = True
                fidelity = _fidelity_for_mask(
                    state["x"],
                    state["raw_expensive"],
                    trial,
                    state["teacher_top"],
                    target_k,
                )
                utility = max(state["base_fidelity"], fidelity)
                delta += utility - state["utility"]
                updates.append((local_index, position, utility))
            if delta <= 1e-12:
                continue
            candidate = (delta / cost, delta, -cost, -corpus_index, updates)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        if best is None:
            break
        _, _, negative_cost, negative_index, updates = best
        corpus_index = -negative_index
        cost = -negative_cost
        selected.add(corpus_index)
        used += cost
        path.append((corpus_index, used))
        for local_index, position, utility in updates:
            states[local_index]["observed"][position] = True
            states[local_index]["utility"] = utility

    plans = {}
    for fraction in budget_fractions:
        limit = float(fraction) * total_cost
        plans[float(fraction)] = {
            corpus_index for corpus_index, cumulative in path if cumulative <= limit
        }
    return plans


def analyze_workload_compiler(
    cube: ScoreCube,
    *,
    base_route: str,
    expensive_route: str,
    candidate_k: int = 20,
    target_k: int = 5,
    target_metric: str = "ndcg_at_5",
    build_costs: Sequence[float] | None = None,
    budget_fractions: Sequence[float] = (0.1, 0.2, 0.4),
    anchors: int = 3,
) -> dict:
    """Compile fit-workload states, then evaluate with three online anchors."""

    cube.validate()
    base = cube.scores[base_route]
    expensive = cube.scores[expensive_route]
    costs = (
        np.ones(len(cube.corpus_ids), dtype=np.float64)
        if build_costs is None
        else np.asarray(build_costs, dtype=np.float64)
    )
    if costs.shape != (len(cube.corpus_ids),) or np.any(costs <= 0):
        raise ValueError("build costs must be positive and corpus-aligned")
    fit_indices = np.flatnonzero(np.asarray(cube.split_roles) == "fit")
    eval_indices = np.flatnonzero(np.asarray(cube.split_roles) == "eval")
    certificates = [
        greedy_certificate(
            base[index],
            expensive[index],
            candidate_k=candidate_k,
            target_k=target_k,
            anchors=anchors,
            objective="order",
        )
        for index in range(len(cube.query_ids))
    ]
    candidate_frequency = np.zeros(len(cube.corpus_ids), dtype=np.float64)
    certificate_frequency = np.zeros(len(cube.corpus_ids), dtype=np.float64)
    marginal_fidelity = np.zeros(len(cube.corpus_ids), dtype=np.float64)
    for query_index in fit_indices:
        certificate = certificates[query_index]
        candidate_frequency[certificate.candidate_indices] += 1.0
        certificate_frequency[
            certificate.candidate_indices[certificate.observed_positions]
        ] += 1.0
        x = _zscore(base[query_index, certificate.candidate_indices])
        raw_expensive = expensive[query_index, certificate.candidate_indices]
        anchor_mask = np.zeros(candidate_k, dtype=bool)
        anchor_mask[_initial_order(candidate_k)[:anchors]] = True
        anchor_ranked = np.lexsort(
            (
                np.arange(candidate_k),
                -_completed_values(x, raw_expensive, anchor_mask),
            )
        )[:target_k]
        anchor_fidelity = _teacher_order_score(
            anchor_ranked, certificate.teacher_top_positions
        )
        for position in np.flatnonzero(~anchor_mask):
            trial = anchor_mask.copy()
            trial[position] = True
            ranked = np.lexsort(
                (np.arange(candidate_k), -_completed_values(x, raw_expensive, trial))
            )[:target_k]
            delta = _teacher_order_score(
                ranked, certificate.teacher_top_positions
            ) - anchor_fidelity
            marginal_fidelity[int(certificate.candidate_indices[position])] += max(
                float(delta), 0.0
            )
    teacher_surface = np.stack(
        [
            _completed_rerank_scores(
                base[index],
                certificate.candidate_indices,
                _zscore(base[index, certificate.candidate_indices])
                + _zscore(expensive[index, certificate.candidate_indices]),
            )
            for index, certificate in enumerate(certificates)
        ]
    )
    base_values = query_metrics(base, cube.relevance, ks=(target_k,))[target_metric]
    teacher_values = query_metrics(
        teacher_surface, cube.relevance, ks=(target_k,)
    )[target_metric]
    total_cost = float(costs.sum())
    greedy_plans = _greedy_listwise_plans(
        certificates,
        base,
        expensive,
        fit_indices,
        costs,
        total_cost=total_cost,
        budget_fractions=budget_fractions,
        candidate_k=candidate_k,
        target_k=target_k,
        anchors=anchors,
    )
    policies = {}
    for fraction in budget_fractions:
        compiled_plans = (
            (
                "candidate_frequency",
                _budgeted_documents(
                    candidate_frequency, costs, fraction * total_cost
                ),
            ),
            (
                "certificate_frequency",
                _budgeted_documents(
                    certificate_frequency, costs, fraction * total_cost
                ),
            ),
            (
                "marginal_fidelity",
                _budgeted_documents(marginal_fidelity, costs, fraction * total_cost),
            ),
            ("greedy_listwise", greedy_plans[float(fraction)]),
        )
        for policy, compiled in compiled_plans:
            fit_fidelities = []
            fit_position_agreements = []
            for query_index in fit_indices:
                certificate = certificates[query_index]
                observed = np.asarray(
                    [int(index) in compiled for index in certificate.candidate_indices],
                    dtype=bool,
                )
                observed[_initial_order(candidate_k)[:anchors]] = True
                x = _zscore(base[query_index, certificate.candidate_indices])
                completed = _completed_values(
                    x,
                    expensive[query_index, certificate.candidate_indices],
                    observed,
                )
                ranked = np.lexsort((np.arange(candidate_k), -completed))[:target_k]
                fit_fidelities.append(
                    _teacher_order_score(ranked, certificate.teacher_top_positions)
                )
                fit_position_agreements.append(
                    float(np.mean(ranked == certificate.teacher_top_positions))
                )
            physical_documents = set(compiled)
            surfaces = []
            gated_surfaces = []
            accepted = []
            agreements = []
            for query_index in eval_indices:
                certificate = certificates[query_index]
                observed = np.asarray(
                    [int(index) in compiled for index in certificate.candidate_indices],
                    dtype=bool,
                )
                observed[_initial_order(candidate_k)[:anchors]] = True
                physical_documents.update(
                    int(certificate.candidate_indices[position])
                    for position in np.flatnonzero(observed)
                )
                x = _zscore(base[query_index, certificate.candidate_indices])
                completed = _completed_values(
                    x,
                    expensive[query_index, certificate.candidate_indices],
                    observed,
                )
                surfaces.append(
                    _completed_rerank_scores(
                        base[query_index], certificate.candidate_indices, completed
                    )
                )
                ranked = np.lexsort((np.arange(candidate_k), -completed))[:target_k]
                stable = True
                for position in np.flatnonzero(observed):
                    leave_one_out = observed.copy()
                    leave_one_out[position] = False
                    if int(leave_one_out.sum()) < 2:
                        continue
                    alternative = _completed_values(
                        x,
                        expensive[query_index, certificate.candidate_indices],
                        leave_one_out,
                    )
                    alternative_ranked = np.lexsort(
                        (np.arange(candidate_k), -alternative)
                    )[:target_k]
                    if set(int(value) for value in alternative_ranked) != set(
                        int(value) for value in ranked
                    ):
                        stable = False
                        break
                accepted.append(stable)
                gated_surfaces.append(surfaces[-1] if stable else base[query_index])
                agreements.append(
                    float(np.mean(ranked == certificate.teacher_top_positions))
                )
            values = query_metrics(
                np.stack(surfaces),
                tuple(cube.relevance[index] for index in eval_indices),
                ks=(target_k,),
            )[target_metric]
            gated_values = query_metrics(
                np.stack(gated_surfaces),
                tuple(cube.relevance[index] for index in eval_indices),
                ks=(target_k,),
            )[target_metric]
            base_eval = base_values[eval_indices]
            teacher_eval = teacher_values[eval_indices]
            full_gain = float((teacher_eval - base_eval).mean())
            gain = float((values - base_eval).mean())
            gated_gain = float((gated_values - base_eval).mean())
            key = f"{policy}_f{fraction:g}"
            policies[key] = {
                "offline_budget_fraction": float(fraction),
                "offline_compiled_documents": len(compiled),
                "offline_compiled_cost": float(costs[list(compiled)].sum()),
                "offline_compiled_cost_fraction": float(
                    costs[list(compiled)].sum() / total_cost
                ),
                "fit_teacher_order_fidelity": float(np.mean(fit_fidelities)),
                "fit_exact_position_agreement": float(
                    np.mean(fit_position_agreements)
                ),
                "total_documents_after_eval_anchors": len(physical_documents),
                "total_cost_after_eval_anchors": float(
                    costs[list(physical_documents)].sum()
                ),
                "total_cost_fraction_after_eval_anchors": float(
                    costs[list(physical_documents)].sum() / total_cost
                ),
                "quality": float(values.mean()),
                "stability_gate": {
                    "acceptance_rate": float(np.mean(accepted)),
                    "quality": float(gated_values.mean()),
                    "full_fusion_gain_recovery": (
                        gated_gain / full_gain if abs(full_gain) > 1e-12 else 0.0
                    ),
                    "vs_teacher": paired_bootstrap_ci(gated_values, teacher_eval),
                },
                "full_fusion_gain_recovery": (
                    gain / full_gain if abs(full_gain) > 1e-12 else 0.0
                ),
                "mean_exact_position_agreement": float(np.mean(agreements)),
                "vs_teacher": paired_bootstrap_ci(values, teacher_eval),
            }
    selected_plans = {}
    for fraction in budget_fractions:
        suffix = f"_f{fraction:g}"
        candidates = {
            key: value for key, value in policies.items() if key.endswith(suffix)
        }
        selected_key = max(
            candidates,
            key=lambda key: (
                candidates[key]["fit_teacher_order_fidelity"],
                -candidates[key]["offline_compiled_cost"],
                key,
            ),
        )
        selected_plans[f"f{fraction:g}"] = {
            "selected_policy": selected_key,
            **candidates[selected_key],
        }
    return {
        "schema_version": 1,
        "queries": len(cube.query_ids),
        "fit_queries": len(fit_indices),
        "eval_queries": len(eval_indices),
        "candidate_k": candidate_k,
        "target_k": target_k,
        "target_metric": target_metric,
        "base_route": base_route,
        "expensive_route": expensive_route,
        "plan_uses_qrels": False,
        "plan_uses_fit_teacher_surface": True,
        "eval_base": float(base_values[eval_indices].mean()),
        "eval_full_teacher": float(teacher_values[eval_indices].mean()),
        "full_corpus_build_cost": total_cost,
        "policies": policies,
        "selected_by_fit_teacher_fidelity": selected_plans,
    }
