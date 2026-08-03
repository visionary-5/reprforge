import numpy as np
import pytest

from reprforge.physical_cost import AtomicMaterializationCostModel
from reprforge.reusable_pair_probe import (
    FrozenPairScoreProvider,
    build_probe_edges,
    normal_boundary_risk,
    select_reusable_pair_probes,
)


def _unit_cost() -> AtomicMaterializationCostModel:
    return AtomicMaterializationCostModel(
        batch_size=4,
        setup_ms=0.0,
        page_ms=1.0,
        batch_ms=0.0,
        score_event_ms=0.0,
    )


def test_frozen_provider_rejects_unmaterialized_score_reads() -> None:
    candidates = np.asarray([[1, 2, 3]])
    visual = np.asarray([[0.1, 0.2, 0.3]])
    provider = FrozenPairScoreProvider(candidates, visual)

    with pytest.raises(RuntimeError, match="unmaterialized"):
        provider.score(0, 1)
    provider.materialize({2})
    assert provider.score(0, 1) == 0.2
    assert provider.score_reads == frozenset({(0, 1)})


def test_reusable_probe_materializes_an_anchor_once() -> None:
    candidates = np.asarray(
        [
            [10, 11, 12],
            [20, 11, 13],
        ]
    )
    locator = np.asarray(
        [
            [2.0, 1.0, 0.95],
            [2.0, 1.0, 0.90],
        ]
    )
    visual = np.asarray(
        [
            [0.0, 0.2, 1.2],
            [0.0, 0.1, 0.9],
        ]
    )
    provider = FrozenPairScoreProvider(candidates, visual)
    plan = select_reusable_pair_probes(
        candidates,
        locator,
        provider,
        _unit_cost(),
        cutoff=2,
        time_budget_ms=3.0,
    )

    assert plan.selected_pages == frozenset({11, 12, 13})
    assert plan.iterations == 2
    assert plan.materialization_rounds == 2
    assert plan.observed_pair_count == 2
    assert provider.materialized_pages == plan.selected_pages
    assert provider.materialization_calls == 2
    assert len(provider.score_reads) == 4


def test_probe_round_batches_actions_before_observation() -> None:
    candidates = np.asarray(
        [[10, 11, 12], [20, 11, 13], [30, 11, 14]]
    )
    locator = np.asarray(
        [[2.0, 1.0, 0.95], [2.0, 1.0, 0.9], [2.0, 1.0, 0.85]]
    )
    visual = np.asarray(
        [[0.0, 0.2, 1.2], [0.0, 0.1, 0.9], [0.0, 0.3, 0.8]]
    )
    provider = FrozenPairScoreProvider(candidates, visual)
    plan = select_reusable_pair_probes(
        candidates,
        locator,
        provider,
        _unit_cost(),
        cutoff=2,
        time_budget_ms=4.0,
        round_page_limit=4,
        empirical_updates=False,
    )

    assert plan.selected_pages == frozenset({11, 12, 13, 14})
    assert plan.materialization_rounds == 1
    assert provider.materialization_calls == 1


def test_probe_edges_and_normal_prior_are_interpretable() -> None:
    candidates = np.asarray([[1, 2, 3, 4]])
    locator = np.asarray([[2.0, 1.0, 0.9, 0.0]])
    edges = build_probe_edges(candidates, locator, cutoff=2)

    assert [(edge.incumbent_page, edge.challenger_page) for edge in edges] == [
        (2, 3),
        (2, 4),
    ]
    assert normal_boundary_risk(edges[0].locator_margin) > normal_boundary_risk(
        edges[1].locator_margin
    )
