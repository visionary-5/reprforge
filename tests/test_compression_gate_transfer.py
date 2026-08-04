import pytest

from reprforge.compression_gate_transfer import summarize_gate_transfer


def _record(dataset, candidate, certificate, safety, fraction, regret):
    return {
        "dataset": dataset,
        "candidate": candidate,
        "certificate_passes": certificate,
        "safety_passes": safety,
        "resident_fraction": fraction,
        "mean_ndcg10_regret": regret,
    }


def test_transfer_summary_compares_selector_with_best_fixed_safe_state():
    report = summarize_gate_transfer(
        [
            _record("a", "pool4", True, True, 0.25, 0.001),
            _record("a", "pool9", False, False, 0.11, 0.03),
            _record("b", "pool4", False, False, 0.25, 0.02),
            _record("b", "pool9", False, False, 0.11, 0.04),
        ]
    )
    assert report["decision_accuracy"] == 1.0
    assert report["false_safe"] == 0
    assert report["false_reject"] == 0
    assert report["all_selected_states_safe"]
    assert report["macro_selected_resident_fraction"] == pytest.approx(0.625)
    assert report["best_fixed_safe_state"] == {
        "candidate": "full",
        "macro_resident_fraction": 1.0,
    }


def test_transfer_summary_counts_false_safe_and_false_reject():
    report = summarize_gate_transfer(
        [
            _record("a", "pool4", True, False, 0.25, 0.03),
            _record("a", "pool9", False, True, 0.11, 0.0),
        ]
    )
    assert report["false_safe"] == 1
    assert report["false_reject"] == 1
    assert report["decision_accuracy"] == 0.0
    assert not report["all_selected_states_safe"]
