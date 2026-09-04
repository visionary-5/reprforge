from pathlib import Path

import numpy as np
import pytest

from reprforge import (
    COMPONENTS,
    LateInteractionIndex,
    VersionManifest,
    load_index,
    maxsim_score,
    save_index,
    target_agreement,
)


def version(**overrides: str) -> VersionManifest:
    base = {name: f"{name}-v1" for name in COMPONENTS}
    base.update(overrides)
    return VersionManifest(**base)


def test_maxsim_and_search_order() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    index = LateInteractionIndex(
        [
            ("a", [[1.0, 0.0], [0.2, 0.8]]),
            ("b", [[0.9, 0.1], [0.0, 1.0]]),
            ("c", [[-1.0, 0.0], [0.0, -1.0]]),
        ]
    )

    assert [row.item_id for row in index.search(query, top_k=2)] == ["b", "a"]
    assert maxsim_score(query, [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(2.0)


def test_target_agreement_is_top_k_overlap() -> None:
    reference = ["a", "b", "c", "d"]
    assert target_agreement(reference, ["a", "b", "c", "d"], k=4) == 1.0
    assert target_agreement(reference, ["d", "c", "x", "y"], k=4) == 0.5
    assert target_agreement(reference, ["x"], k=1) == 0.0
    with pytest.raises(ValueError):
        target_agreement(reference, reference, k=0)


def test_index_round_trip_records_version_and_source(tmp_path: Path) -> None:
    root = tmp_path / "index"
    index = LateInteractionIndex([("a", np.eye(3)), ("b", [[1.0, 1.0, 0.0]])])

    written = save_index(root, index, version(), source="post_vision_pca256")
    loaded, observed = load_index(root)

    assert observed == written
    assert observed.source == "post_vision_pca256"
    assert observed.version == version()
    assert loaded.item_ids == index.item_ids
    assert loaded.vector_count == index.vector_count
    with pytest.raises(FileExistsError):
        save_index(root, index, version())


def test_index_rejects_mutated_payload(tmp_path: Path) -> None:
    root = tmp_path / "index"
    save_index(root, LateInteractionIndex([("a", np.eye(2))]), version())
    with (root / "vectors.npz").open("ab") as payload:
        payload.write(b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        load_index(root)


def test_index_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LateInteractionIndex([])
    with pytest.raises(ValueError, match="unique"):
        LateInteractionIndex([("a", np.eye(2)), ("a", np.eye(2))])
    with pytest.raises(ValueError, match="one dimension"):
        LateInteractionIndex([("a", np.eye(2)), ("b", np.eye(3))])
