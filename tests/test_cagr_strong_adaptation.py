import hashlib
import json

import numpy as np

from tools.analyze_cagr_strong_adaptation import _finance_gate, load_access_graph


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_access_graph_loader_opens_only_bm25_runtime(tmp_path):
    bm25 = tmp_path / "bm25"
    bm25.mkdir()
    runtime = bm25 / "runtime.npz"
    np.savez(
        runtime,
        scores=np.asarray([[0.2, 0.9], [0.8, 0.1]]),
        corpus_ids=np.asarray(["p0", "p1"]),
        query_ids=np.asarray(["q0", "q1"]),
    )
    manifest = bm25 / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"runtime_file": runtime.name, "runtime_sha256": _sha256(runtime)}
        )
    )
    # Poison files represent surfaces forbidden during HR selection.  Their
    # presence must neither be required nor inspected by load_access_graph.
    (tmp_path / "oracle-labels.json").write_text("not-json")
    (tmp_path / "visual-runtime.npz").write_text("not-an-npz")

    loaded = load_access_graph(tmp_path, candidate_k=1)

    assert loaded["cohorts"] == [[1], [0]]
    assert loaded["candidate_union_pages"] == 2
    assert loaded["provenance"]["oracle_or_visual_files_opened"] is False
    assert loaded["provenance"]["opened_files"] == [
        str(manifest.resolve()),
        str(runtime.resolve()),
    ]


def _aggregate(page, cost):
    return {
        "completion_pages": {"mean": page, "p95": page},
        "completion_unit_cost": {"mean": cost, "p95": cost},
        "normalized_quality_regret_auc": {"mean": 0.2},
        "starvation": {"fraction": 0.0},
    }


def test_finance_gate_stops_when_either_frontier_advantage_is_below_five_percent():
    config = {
        "family": "threshold",
        "theta": 0.1,
        "group_pool": 20,
        "capacity": 80,
        "target_group_size": 8,
    }
    finance = {}
    for model in ("burst", "poisson"):
        finance[model] = {
            "hr_selected_lower_theta": {"aggregate": _aggregate(100.0, 100.0)},
            # Page advantage is 10%, but cost advantage is only 4%.
            "frontier_for_lower_theta": {"aggregate": _aggregate(90.0, 96.0)},
        }

    gate = _finance_gate(finance, {"lower_theta": config})

    assert gate["decision"] == "STOP/DOWNGRADE"
    assert all(not check["passes"] for check in gate["checks"])
    assert all(
        np.isclose(check["frontier_unit_cost_advantage"], 0.04)
        for check in gate["checks"]
    )


def test_finance_gate_stops_explicitly_when_hr_has_no_deployable_candidate():
    gate = _finance_gate({}, {"lower_theta": None, "fixed_size": None})

    assert gate["decision"] == "STOP/DOWNGRADE"
    assert gate["checks"] == []
    assert gate["no_deployable_hr_selection"] is True
    assert "no adaptation passed" in gate["paper_action"]
