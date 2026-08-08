from __future__ import annotations

from pathlib import Path

import pytest

from tools.analyze_physical_static_materialization_v0 import _external_full_case


def test_external_full_case_binds_ranking_cost_and_evidence(tmp_path: Path) -> None:
    ranking = tmp_path / "ranking.txt"
    ranking.write_text("q1\td1\t1.0\nq1\td2\t0.5\n")
    evidence = tmp_path / "shard-manifest.json"
    evidence.write_text("{}\n")

    result = _external_full_case(
        ranking,
        build_wall_seconds=12.5,
        index_bytes=1234,
        evidence=[evidence],
    )

    assert result["rankings"] == {"q1": ["d1", "d2"]}
    assert result["build_wall_seconds"] == 12.5
    assert result["index_bytes"] == 1234
    assert list(result["evidence_sha256"]) == [str(evidence.resolve())]


def test_external_full_case_requires_provenance(tmp_path: Path) -> None:
    ranking = tmp_path / "ranking.txt"
    ranking.write_text("q1\td1\t1.0\n")
    with pytest.raises(ValueError, match="provenance"):
        _external_full_case(
            ranking,
            build_wall_seconds=1.0,
            index_bytes=1,
            evidence=[],
        )
