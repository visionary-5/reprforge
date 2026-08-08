import numpy as np

from reprforge.partial_vlm_materialization import ScoreSurface
from reprforge.value_aware_materialization import (
    CompilerConfig,
    anchor_rank_ranking,
    calibrated_ranking,
    cheap_history_features,
    compile_value_aware_index,
    typed_materialization_ranking,
)


def _surface(
    *,
    text_scores: np.ndarray,
    visual_scores: np.ndarray,
    qrels: np.ndarray,
) -> ScoreSurface:
    queries, pages = text_scores.shape
    return ScoreSurface(
        name="fixture",
        query_ids=np.asarray([f"q{i}" for i in range(queries)]),
        corpus_ids=np.asarray([f"p{i}" for i in range(pages)]),
        text_scores=np.asarray(text_scores, dtype=np.float64),
        visual_scores=np.asarray(visual_scores, dtype=np.float64),
        qrels=np.asarray(qrels, dtype=np.float64),
        text_bytes=np.arange(1, pages + 1, dtype=np.float64),
        visual_bytes=np.full(pages, 10.0),
        visual_encode_ms=np.ones(pages, dtype=np.float64),
        input_sha256={},
    )


def test_calibrated_ranking_ignores_unmaterialized_visual_columns() -> None:
    surface = _surface(
        text_scores=np.asarray([[4.0, 3.0, 2.0, 1.0, 0.0]]),
        visual_scores=np.asarray([[0.0, 2.0, 0.0, 1.0, 100.0]]),
        qrels=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    ranking = calibrated_ranking(
        surface,
        0,
        admitted_pages=[1],
        anchor_pages=[2, 3],
        calibration_quantile=0.9,
        visual_weight=1.0,
    )
    changed_visual = surface.visual_scores.copy()
    changed_visual[0, 4] = 1_000_000.0
    changed = _surface(
        text_scores=surface.text_scores,
        visual_scores=changed_visual,
        qrels=surface.qrels,
    )
    changed_ranking = calibrated_ranking(
        changed,
        0,
        admitted_pages=[1],
        anchor_pages=[2, 3],
        calibration_quantile=0.9,
        visual_weight=1.0,
    )
    assert ranking.tolist() == changed_ranking.tolist()


def test_below_background_visual_score_abstains() -> None:
    surface = _surface(
        text_scores=np.asarray([[4.0, 3.0, 2.0, 1.0]]),
        visual_scores=np.asarray([[0.0, 0.5, 0.0, 1.0]]),
        qrels=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    ranking = calibrated_ranking(
        surface,
        0,
        admitted_pages=[1],
        anchor_pages=[2, 3],
        calibration_quantile=0.9,
        visual_weight=10.0,
    )
    assert ranking.tolist() == surface.text_order[0].tolist()


def test_anchor_rank_does_not_treat_subset_rank_as_global_rank() -> None:
    surface = _surface(
        text_scores=np.asarray([[4.0, 3.0, 2.0, 1.0]]),
        visual_scores=np.asarray([[0.0, 0.5, 0.0, 1.0]]),
        qrels=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    ranking = anchor_rank_ranking(
        surface,
        0,
        admitted_pages=[1],
        anchor_pages=[2, 3],
        visual_top_k=1,
    )
    assert ranking.tolist() == surface.text_order[0].tolist()


def test_benefit_page_cannot_escape_text_candidate_scope() -> None:
    surface = _surface(
        text_scores=np.asarray([[4.0, 3.0, 2.0, 1.0, 0.0]]),
        visual_scores=np.asarray([[0.0, 0.0, 0.0, 1.0, 100.0]]),
        qrels=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    ranking = typed_materialization_ranking(
        surface,
        0,
        benefit_pages=[4],
        coverage_pages=[],
        anchor_pages=[2, 3],
        candidate_k=2,
        benefit_weight=100.0,
    )
    assert ranking.tolist() == surface.text_order[0].tolist()


def test_coverage_page_can_repair_candidate_escape() -> None:
    surface = _surface(
        text_scores=np.asarray([[4.0, 3.0, 2.0, 1.0, 0.0]]),
        visual_scores=np.asarray([[0.0, 0.0, 0.0, 1.0, 100.0]]),
        qrels=np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    ranking = typed_materialization_ranking(
        surface,
        0,
        benefit_pages=[],
        coverage_pages=[4],
        anchor_pages=[2, 3],
        candidate_k=2,
        coverage_quantile=0.99,
        coverage_weight=1.0,
    )
    assert ranking[0] == 4


def test_cheap_features_do_not_read_visual_scores_or_qrels() -> None:
    text = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    original = _surface(
        text_scores=text,
        visual_scores=np.zeros_like(text),
        qrels=np.eye(2, 4),
    )
    mutated = _surface(
        text_scores=text,
        visual_scores=np.full_like(text, 999.0),
        qrels=np.ones_like(text),
    )
    np.testing.assert_allclose(
        cheap_history_features(original, [0, 1]),
        cheap_history_features(mutated, [0, 1]),
    )


def test_compiler_observes_signed_value_and_respects_budget() -> None:
    queries, pages = 3, 6
    text = np.tile(np.asarray([5.0, 6.0, 4.0, 3.0, 2.0, 1.0]), (queries, 1))
    visual = np.zeros((queries, pages), dtype=np.float64)
    visual[:, 0] = 10.0
    qrels = np.zeros((queries, pages), dtype=np.float64)
    qrels[:, 0] = 1.0
    surface = _surface(text_scores=text, visual_scores=visual, qrels=qrels)
    result = compile_value_aware_index(
        surface,
        [0, 1, 2],
        page_costs=np.ones(pages),
        maximum_cost=6.0,
        config=CompilerConfig(
            anchor_pages=2,
            calibration_quantile=0.75,
            visual_weight=10.0,
            exploration=1.0,
            seed=3,
        ),
    )
    assert result["spent_cost"] <= 6.0
    assert set(result["admitted_pages"]) <= set(result["probed_pages"])
    assert 0 in result["admitted_pages"]
    assert result["rejected_pages"]
    assert any(row["observed_marginal_ndcg_at_10"] > 0 for row in result["trace"])


def test_compiler_selection_is_independent_of_future_rows() -> None:
    text = np.asarray(
        [
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0],
            [4.0, 5.0, 3.0, 6.0, 2.0, 7.0, 1.0, 8.0],
        ]
    )
    visual = text[:, ::-1].copy()
    qrels = np.zeros_like(text)
    qrels[0, 1] = 1.0
    qrels[1, 6] = 1.0
    qrels[2, 2] = 1.0
    qrels[3, 5] = 1.0
    original = _surface(text_scores=text, visual_scores=visual, qrels=qrels)
    changed_visual = visual.copy()
    changed_qrels = qrels.copy()
    changed_visual[2:] = np.arange(16, dtype=np.float64).reshape(2, 8) * 1000.0
    changed_qrels[2:] = 1.0 - changed_qrels[2:]
    changed = _surface(
        text_scores=text,
        visual_scores=changed_visual,
        qrels=changed_qrels,
    )
    kwargs = {
        "history_queries": [0, 1],
        "page_costs": np.ones(8),
        "maximum_cost": 5.0,
        "config": CompilerConfig(anchor_pages=2, seed=11),
    }
    first = compile_value_aware_index(original, **kwargs)
    second = compile_value_aware_index(changed, **kwargs)
    for key in ("anchor_pages", "probed_pages", "admitted_pages", "rejected_pages"):
        assert first[key] == second[key]
