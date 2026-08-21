from pathlib import Path

import numpy as np
import pytest

from reprforge import (
    BackboneProfile,
    CompactIndex,
    VersionManifest,
    load_index,
    maxsim_score,
    plan_topology_anchored,
    refine_candidates,
    save_index,
)


def test_compact_search_then_full_refinement() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    index = CompactIndex(
        [
            ("a", [[1.0, 0.0], [0.2, 0.8]]),
            ("b", [[0.9, 0.1], [0.0, 1.0]]),
            ("c", [[-1.0, 0.0], [0.0, -1.0]]),
        ]
    )
    candidates = index.search(query, top_k=2)
    full = {
        "a": np.asarray([[1.0, 0.0], [0.1, 0.2]]),
        "b": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }

    refined = refine_candidates(index, query, candidates, full.__getitem__)

    assert [row.item_id for row in candidates] == ["b", "a"]
    assert [row.item_id for row in refined] == ["b", "a"]
    assert maxsim_score(query, full["b"]) == pytest.approx(2.0)


def test_index_artifact_round_trip_records_compile_plan(tmp_path: Path) -> None:
    root = tmp_path / "index"
    index = CompactIndex(
        [("a", np.eye(3)), ("b", np.asarray([[1.0, 1.0, 0.0]]))]
    )
    profile = BackboneProfile("test", 18, 6, 16, 8)
    plan = plan_topology_anchored(profile, grid_shape=(4, 4))

    written = save_index(root, index, plan)
    loaded, observed = load_index(root)

    assert observed == written
    assert observed.plan.fingerprint == plan.fingerprint
    assert loaded.item_ids == index.item_ids
    assert loaded.vector_count == index.vector_count


def test_versioned_index_round_trip_and_dependency_diff(tmp_path: Path) -> None:
    root = tmp_path / "versioned-index"
    index = CompactIndex([("a", np.eye(2))])
    profile = BackboneProfile("test", 18, 6, 16, 8)
    plan = plan_topology_anchored(profile, grid_shape=(4, 4))
    old = VersionManifest(
        source="pages-v1",
        processor="processor-v1",
        vision="vision-v1",
        base_embedding="embedding-v1",
        adapter="adapter-v1",
        projection="projection-v1",
        index_policy="policy-v1",
    )
    target = VersionManifest(
        source="pages-v1",
        processor="processor-v1",
        vision="vision-v1",
        base_embedding="embedding-v1",
        adapter="adapter-v2",
        projection="projection-v2",
        index_policy="policy-v1",
    )

    written = save_index(root, index, plan, old)
    _, observed = load_index(root)
    update = observed.version.update_scenario(target, "domain_adapter")

    assert written.format_version == 2
    assert observed.version == old
    assert update.changed_components == frozenset({"adapter", "projection"})


def test_identical_versions_do_not_create_an_update() -> None:
    version = VersionManifest(*("same",) * 7)

    with pytest.raises(ValueError, match="identical"):
        version.update_scenario(version, "noop")


def test_version_manifest_rejects_non_string_fingerprints() -> None:
    value = VersionManifest(*("valid",) * 6, index_policy=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="index_policy"):
        value.validate()


def test_index_artifact_rejects_mutated_payload(tmp_path: Path) -> None:
    root = tmp_path / "index"
    index = CompactIndex([("a", np.eye(2))])
    profile = BackboneProfile("test", 18, 6, 16, 8)
    plan = plan_topology_anchored(profile, grid_shape=(4, 4))
    save_index(root, index, plan)

    with (root / "vectors.npz").open("ab") as payload:
        payload.write(b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        load_index(root)


def test_refinement_rejects_items_outside_compact_locator() -> None:
    index = CompactIndex([("a", [[1.0, 0.0]])])
    with pytest.raises(KeyError, match="outside the compact index"):
        refine_candidates(
            index,
            [[1.0, 0.0]],
            ["missing"],
            lambda _: [[1.0, 0.0]],
        )
