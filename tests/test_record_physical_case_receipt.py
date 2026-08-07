from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.record_physical_case_receipt import build_receipt


def _case(tmp_path: Path) -> Path:
    root = tmp_path / "random-005"
    (root / "timing").mkdir(parents=True)
    (root / "full/index").mkdir(parents=True)
    (root / "full/result").mkdir(parents=True)
    (root / "run-manifest.json").write_text(json.dumps({"test": True}))
    (root / "timing/full-build.time").write_text("real 1.25\nuser 1\nsys 0\n")
    (root / "timing/full-eval.time").write_text("real 2.5\nuser 2\nsys 0\n")
    (root / "full/result/ranking.txt").write_text(
        "q1\td1\t1.0\nq1\td2\t0.5\n"
    )
    (root / "full/build.log").write_text("built\n")
    (root / "full/eval.log").write_text("evaluated\n")
    (root / "full/index/embeddings.pt").write_bytes(b"1234")
    return root


def test_receipt_records_releasable_evidence(tmp_path: Path) -> None:
    root = _case(tmp_path)
    receipt = build_receipt(root, None)
    assert receipt["status"] == "complete_and_safe_to_release_reproducible_index"
    assert receipt["physical_index"] == {"bytes": 4, "files": 1}
    assert receipt["ranking"] == {"queries": 1, "rows": 2}
    assert receipt["timing_seconds"]["direct_build_wall"] == 1.25


def test_receipt_rejects_symlinks_inside_index(tmp_path: Path) -> None:
    root = _case(tmp_path)
    (root / "outside").write_text("not index data")
    (root / "full/index/link").symlink_to(root / "outside")
    with pytest.raises(ValueError, match="symbolic link"):
        build_receipt(root, None)
