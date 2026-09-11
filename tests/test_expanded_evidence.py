"""Evidence verification must reject missing pages without requiring success."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "expanded_verify",
    Path(__file__).parents[1] / "experiments/full-collection-endpoint/verify.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence(tmp_path):
    rows = [
        {
            "case": "control",
            "index": i,
            "page": {"image_sha256": str(i)},
            "state_sha256": str(i),
            "raw_vision_calls": 1,
            "replay_vision_calls": 0,
            "checks": {"factorized_replay": {"bit_equal": False, "finite": True}},
        }
        for i in range(2)
    ]
    result = {
        "pages_per_case": 2,
        "rows": 2,
        "protocol": {"cases": {"control": "test"}},
        "by_case": {"control": {"factorized_replay": 0}},
        "all_native_called_vision": True,
        "all_replay_skipped_vision": True,
        "all_checks_finite": True,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_negative_results_are_valid_evidence(tmp_path):
    evidence(tmp_path)
    assert (
        module.verify_core(tmp_path, False)["outcomes"]["control"]["factorized_replay"]
        == 0
    )


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_summary"])
def test_incomplete_or_inconsistent_evidence_is_rejected(tmp_path, corruption):
    rows = evidence(tmp_path)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0]["checks"]["factorized_replay"]["bit_equal"] = True
    (tmp_path / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(AssertionError):
        module.verify_core(tmp_path, False)
