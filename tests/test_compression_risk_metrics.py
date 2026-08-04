import numpy as np

from reprforge.compression_risk_metrics import (
    evaluate_compression_pair,
    ranking_fidelity,
    ranking_safety_certificate,
    rbo_ext,
    regret_summary,
    summarize_costs,
)


def test_identical_rankings_have_perfect_fidelity_and_zero_regret():
    descending = np.arange(120, dtype=float)[::-1]
    scores = np.stack((descending, np.roll(descending, 1)))
    relevance = ({0: 1.0}, {1: 1.0})
    report = evaluate_compression_pair(
        scores,
        scores.copy(),
        relevance,
        quality_ks=(1, 2, 3, 5, 10, 100),
        bootstrap_resamples=50,
    )
    assert report["quality"]["ndcg_at_10"]["mean_regret"] == 0.0
    assert report["ranking_fidelity"]["top_10_overlap"]["mean"] == 1.0
    assert np.isclose(
        report["ranking_fidelity"]["rbo_ext_at_100_p0.95"]["mean"], 1.0
    )
    assert np.isclose(
        report["ranking_fidelity"]["top_10_overlap"]["bootstrap"][
            "one_sided_95_lower"
        ],
        1.0,
    )
    assert report["safety_gate"]["passes"]
    assert report["qrel_free_ranking_certificate"]["passes"]


def test_regret_tail_and_catastrophic_harm_are_positive_for_bad_candidate():
    reference = np.asarray([1.0, 1.0, 1.0, 1.0])
    candidate = np.asarray([1.0, 0.9, 0.5, 0.0])
    report = regret_summary(
        reference,
        candidate,
        catastrophic_threshold=0.2,
        resamples=50,
    )
    assert report["mean_regret"] > 0.0
    assert report["worst_5pct_cvar"] == 1.0
    assert report["catastrophic_harm_fraction"] == 0.5


def test_ranking_fidelity_uses_stable_ties_and_detects_top_loss():
    reference = np.asarray([[3.0, 2.0, 1.0, 0.0]])
    candidate = np.asarray([[0.0, 1.0, 2.0, 3.0]])
    fidelity = ranking_fidelity(
        reference,
        candidate,
        top_ks=(1, 2),
        rbo_depth=4,
        rbo_p=0.9,
    )
    assert fidelity["top_1_overlap"].tolist() == [0.0]
    assert fidelity["top_2_overlap"].tolist() == [0.0]
    assert 0.0 <= fidelity["rbo_ext_at_4_p0.9"][0] < 1.0
    assert np.isclose(rbo_ext([0, 1, 2], [0, 1, 2]), 1.0)


def test_cost_summary_requires_paired_token_work():
    report = summarize_costs(
        reference_vector_bytes=100,
        candidate_vector_bytes=25,
        vector_bytes_kind="persistent-bank",
        reference_token_work=1000,
        candidate_token_work=200,
        token_work_kind="document-vectors-per-exhaustive-query",
    )
    assert report["vector_byte_fraction"] == 0.25
    assert report["token_work_fraction"] == 0.2
    assert report["vector_bytes_kind"] == "persistent-bank"


def test_ranking_fidelity_rejects_invalid_depth():
    scores = np.asarray([[1.0, 0.0]])
    try:
        ranking_fidelity(scores, scores, top_ks=(1,), rbo_depth=3)
    except ValueError as error:
        assert "RBO depth" in str(error)
    else:
        raise AssertionError("invalid RBO depth was accepted")


def test_ranking_certificate_rejects_unstable_top10_even_with_candidate_recall():
    fidelity = {
        "top_10_overlap": np.asarray([0.8, 0.9, 0.8, 0.9]),
        "full_top10_retained_at_candidate_top100": np.ones(4),
    }
    certificate = ranking_safety_certificate(fidelity, resamples=50)
    assert not certificate["passes"]
    assert not certificate["uses_qrels"]
