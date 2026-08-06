import json
from pathlib import Path

import pytest

from tools.compare_omni_vidore_pipelines import compare


def _official(root: Path, model: str, domain: str, ndcg: float) -> None:
    path = root / "results" / "metrics" / model / f"vidore_v3_{domain}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "overall": {"ndcg_cut_10": ndcg, "recall_100": 0.8},
                    "timing": {
                        "indexing_throughput_ms_per_doc": 2.0,
                        "search_throughput_ms_per_query": 3.0,
                    },
                }
            }
        )
    )


def _omni() -> dict:
    return {
        "analysis_scope": {"queries": 2, "corpus_pages": 10},
        "per_query": [
            {
                "ndcg_at_10": {"full": 0.6, "hpool": 0.5},
                "candidate_recall_at_100": {"hpool": 0.7},
            },
            {
                "ndcg_at_10": {"full": 0.8, "hpool": 0.7},
                "candidate_recall_at_100": {"hpool": 0.9},
            },
        ],
        "multi_locator_cost_ledger": {
            "full_index_bytes": 1000,
            "hpool_index_bytes": 100,
            "interpretation": "test",
        },
    }


def test_compare_extracts_quality_cost_and_provenance(tmp_path: Path):
    official = tmp_path / "official"
    _official(official, "model-a", "hr", 0.65)
    omni_path = tmp_path / "omni.json"
    omni_path.write_text(json.dumps(_omni()))

    result = compare(
        official,
        "abc123",
        ["model-a"],
        {"hr": (omni_path, _omni())},
    )

    domain = result["domains"]["hr"]
    assert domain["omni"]["mean_ndcg_at_10"]["full"] == pytest.approx(0.7)
    assert domain["omni"]["mean_qrel_candidate_recall_at_100"]["hpool"] == 0.8
    assert domain["omni"]["index_bytes"]["full"] == 1000
    assert domain["official_pipelines"][0]["ndcg_at_10"] == 0.65
    assert result["official_source"]["revision"] == "abc123"


def test_compare_rejects_missing_official_artifact(tmp_path: Path):
    omni_path = tmp_path / "omni.json"
    omni_path.write_text(json.dumps(_omni()))
    with pytest.raises(FileNotFoundError):
        compare(
            tmp_path,
            "abc123",
            ["missing"],
            {"hr": (omni_path, _omni())},
        )
