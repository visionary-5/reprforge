from pathlib import Path

import pytest

from tools.summarize_omni_cross_domain_failures import summarize


def _report(queries: int, harm: int, contained: int, recovered: int):
    return {
        "analysis_scope": {"queries": queries, "corpus_pages": 1000},
        "teacher_containment_groups": {"counts": {"20": queries}},
        "quality_preservation_groups": {"counts": {"20": queries}},
        "failure_mechanism_summary": {
            "hpool_quality_harm_queries": harm,
            "hpool_quality_harm_with_full_top10_contained_at_100": contained,
            "hpool_quality_harm_recovered_by_cascade100": recovered,
        },
        "escape_audit": {
            "ranking_escape_queries": 2,
            "teacher_evidence_escape_queries": 1,
            "teacher_evidence_escape_documents": 1,
            "teacher_evidence_escape_documents_recovered_by_agc_at_100": 1,
        },
        "relevant_candidate_boundary": {
            "hpool_mean_recall_at_100": 0.8,
            "union_mean_recall_at_100": 0.9,
            "queries_improved_by_union": 3,
        },
        "observable_risk_signal_audit": {
            "targets": {
                "risk": {
                    "positives": 2,
                    "markers": [
                        {"marker": "free", "oriented_auc": 0.6, "requires_agc": False},
                        {"marker": "agc", "oriented_auc": 0.7, "requires_agc": True},
                    ],
                }
            }
        },
    }


def test_summary_aggregates_failure_mechanisms(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}")
    second.write_text("{}")
    result = summarize(
        {
            "first": (first, _report(10, 4, 3, 4)),
            "second": (second, _report(20, 6, 5, 5)),
        }
    )

    assert result["aggregate"]["queries"] == 30
    assert result["aggregate"]["contained_fraction_among_hpool_harms"] == 0.8
    assert result["aggregate"]["cascade100_recovery_fraction_among_hpool_harms"] == 0.9
    assert result["aggregate"]["query_weighted_union_relevant_recall_at_100"] == pytest.approx(0.9)
    assert result["domains"]["first"]["risk_markers"]["risk"]["best_with_agc"]["marker"] == "agc"


def test_summary_requires_multiple_domains(tmp_path: Path):
    path = tmp_path / "only.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="at least two"):
        summarize({"only": (path, _report(10, 1, 1, 1))})
