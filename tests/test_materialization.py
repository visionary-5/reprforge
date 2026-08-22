import pytest

from reprforge import (
    MaterializationOption,
    UpdateScenario,
    choose_materializations,
    evaluate_materializations,
)


def test_measured_colqwen_update_selects_pca_ir() -> None:
    pages = 20_395
    decision = choose_materializations(
        (
            MaterializationOption(
                name="pca256_post_vision",
                depends_on=frozenset({"processor", "vision", "base_embedding"}),
                storage_bytes=6_266_593_554,
                replay_seconds=60.849242073331475 * pages / 1000,
                materialization_seconds=29.750906523666345,
                quality_fraction=0.999,
            ),
        ),
        (
            UpdateScenario(
                "decoder_adapter_and_projection",
                frozenset({"adapter", "projection"}),
                expected_count=2,
            ),
        ),
        raw_rebuild_seconds=245.16711452873466 * pages / 1000,
        storage_budget_bytes=6_369_873_920,
    )

    assert decision.selected == ("pca256_post_vision",)
    assert decision.routes[0].source == "pca256_post_vision"
    assert decision.saving_fraction > 0.7


def test_measured_colpali_one_update_rejects_large_low_payoff_ir() -> None:
    decision = choose_materializations(
        (
            MaterializationOption(
                name="exact_post_vision",
                depends_on=frozenset({"processor", "vision", "base_embedding"}),
                storage_bytes=212_023_360,
                replay_seconds=0.9218522110022604,
                materialization_seconds=0.6972735489835031,
            ),
        ),
        (
            UpdateScenario(
                "colpali_v12_to_v13",
                frozenset({"adapter", "projection"}),
            ),
        ),
        raw_rebuild_seconds=1.133113782008877,
        storage_budget_bytes=212_023_360,
    )

    assert decision.selected == ()
    assert decision.routes[0].source == "raw"
    assert decision.expected_seconds == pytest.approx(1.133113782008877)


def test_invalidated_artifact_falls_back_to_raw() -> None:
    decision = choose_materializations(
        (
            MaterializationOption(
                name="post_vision",
                depends_on=frozenset({"processor", "vision"}),
                storage_bytes=100,
                replay_seconds=2,
            ),
        ),
        (UpdateScenario("vision_upgrade", frozenset({"vision"}), expected_count=3),),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
    )

    assert decision.selected == ()
    assert decision.routes[0].source == "raw"


def test_portfolio_routes_different_updates_from_different_boundaries() -> None:
    decision = choose_materializations(
        (
            MaterializationOption(
                name="post_vision",
                depends_on=frozenset({"processor", "vision"}),
                storage_bytes=80,
                replay_seconds=4,
            ),
            MaterializationOption(
                name="terminal",
                depends_on=frozenset({"processor", "vision", "adapter", "projection"}),
                storage_bytes=20,
                replay_seconds=0,
            ),
        ),
        (
            UpdateScenario("adapter", frozenset({"adapter"})),
            UpdateScenario(
                "index_policy", frozenset({"index_policy"}), expected_count=4
            ),
        ),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
    )

    assert set(decision.selected) == {"post_vision", "terminal"}
    assert [route.source for route in decision.routes] == ["post_vision", "terminal"]
    assert decision.expected_seconds == pytest.approx(4)


def test_quality_and_storage_contracts_can_exclude_an_artifact() -> None:
    option = MaterializationOption(
        name="lossy",
        depends_on=frozenset({"vision"}),
        storage_bytes=101,
        replay_seconds=1,
        quality_fraction=0.98,
    )
    update = UpdateScenario("adapter", frozenset({"adapter"}))

    too_large = choose_materializations(
        (option,),
        (update,),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
        minimum_quality_fraction=0.97,
    )
    too_lossy = choose_materializations(
        (option,),
        (update,),
        raw_rebuild_seconds=10,
        storage_budget_bytes=101,
        minimum_quality_fraction=0.99,
    )

    assert too_large.selected == ()
    assert too_lossy.selected == ()


def test_calibrated_selection_can_be_scored_with_holdout_measurements() -> None:
    update = UpdateScenario("adapter", frozenset({"adapter"}), expected_count=2)
    calibration = MaterializationOption(
        name="post_vision",
        depends_on=frozenset({"processor", "vision"}),
        storage_bytes=100,
        replay_seconds=2,
        materialization_seconds=1,
    )
    selected = choose_materializations(
        (calibration,),
        (update,),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
    )
    holdout = MaterializationOption(
        name="post_vision",
        depends_on=calibration.depends_on,
        storage_bytes=90,
        replay_seconds=3,
        materialization_seconds=2,
    )

    observed = evaluate_materializations(
        selected.selected,
        (holdout,),
        (update,),
        raw_rebuild_seconds=12,
    )

    assert observed.selected == ("post_vision",)
    assert observed.expected_seconds == pytest.approx(8)
    assert observed.raw_baseline_seconds == pytest.approx(24)


def test_validation_cost_can_make_legal_reuse_slower_than_raw() -> None:
    option = MaterializationOption(
        name="post_vision",
        depends_on=frozenset({"vision"}),
        storage_bytes=10,
        replay_seconds=8,
    )
    update = UpdateScenario(
        "adapter",
        frozenset({"adapter"}),
        validation_seconds=3,
    )

    decision = evaluate_materializations(
        ("post_vision",),
        (option,),
        (update,),
        raw_rebuild_seconds=10,
    )

    assert decision.routes[0].source == "raw"
    assert decision.routes[0].seconds_per_update == 10
    assert decision.routes[0].validation_seconds == 0


def test_validation_cost_is_charged_to_a_winning_reuse_route() -> None:
    option = MaterializationOption(
        name="post_vision",
        depends_on=frozenset({"vision"}),
        storage_bytes=10,
        replay_seconds=4,
    )
    update = UpdateScenario(
        "adapter",
        frozenset({"adapter"}),
        expected_count=2,
        validation_seconds=3,
    )

    decision = evaluate_materializations(
        ("post_vision",),
        (option,),
        (update,),
        raw_rebuild_seconds=10,
    )

    assert decision.routes[0].source == "post_vision"
    assert decision.routes[0].seconds_per_update == 7
    assert decision.routes[0].validation_seconds == 3
    assert decision.expected_seconds == 14
