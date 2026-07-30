import json
from pathlib import Path

import numpy as np
import pytest

from reprforge.heterogeneous_index import write_embedding_bank
from reprforge.versioned_visual_index import (
    TieredNumpyRuntime,
    VersionedVisualIndex,
    create_versioned_visual_index,
)


def _embedding(rows):
    return np.asarray(rows, dtype=np.float32)


def _write_bank(path: Path) -> None:
    write_embedding_bank(
        path,
        item_ids=("item-a", "item-b", "item-c"),
        route_embeddings={
            "text": (
                _embedding([[1.0, 0.0]]),
                _embedding([[0.0, 1.0]]),
                _embedding([[0.5, 0.5]]),
            ),
            "image": (
                _embedding([[0.2, 0.0], [0.0, 0.2]]),
                _embedding([[1.0, 0.0], [0.0, 1.0]]),
                _embedding([[0.8, 0.2], [0.2, 0.8]]),
            ),
        },
        query_ids=("query",),
        query_embeddings=(_embedding([[1.0, 0.0], [0.0, 1.0]]),),
    )


def test_materialization_overrides_base_and_cache_hits_are_noops(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_bank(bank)
    root = tmp_path / "tiered"
    manifest = create_versioned_visual_index(
        bank=bank,
        output=root,
        base_route="text",
        visual_route="image",
    )
    index = VersionedVisualIndex(root)
    query = _embedding([[1.0, 0.0], [0.0, 1.0]])

    assert manifest["merge_semantics"] == "active-delta-overrides-base"
    assert index.active_version == 0
    assert index.cached_item_ids == frozenset()
    assert np.allclose(index.runtime().score(query), [1.0, 1.0, 1.0])

    first = index.materialize(["item-b", "item-a"])
    assert first["published"] is True
    assert first["version"] == 1
    assert first["new_items"] == 2
    assert index.cached_item_ids == frozenset({"item-a", "item-b"})
    # item-a's visual score is lower than its text score. Replacement, rather
    # than max(base, delta), is therefore observable in this correctness test.
    assert np.allclose(index.runtime().score(query), [0.4, 2.0, 1.0])

    hit = index.materialize(["item-a", "item-b"])
    assert hit == {
        "version": 1,
        "published": False,
        "requested_items": 2,
        "new_items": 0,
        "cache_hits": 2,
    }
    assert sorted((root / "versions").iterdir()) == [
        root / "versions" / "version-00000001"
    ]


def test_stage_publish_and_rollback_keep_generations_immutable(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_bank(bank)
    root = tmp_path / "tiered"
    create_versioned_visual_index(
        bank=bank,
        output=root,
        base_route="text",
        visual_route="image",
    )
    index = VersionedVisualIndex(root)
    query = _embedding([[1.0, 0.0], [0.0, 1.0]])
    index.materialize(["item-b"])
    version_one_metadata = (
        root / "versions" / "version-00000001" / "version.json"
    ).read_bytes()

    staged = index.stage(["item-c"])
    assert staged is not None
    assert staged.version == 2
    assert staged.parent_version == 1
    assert staged.new_item_ids == ("item-c",)
    assert index.active_version == 1
    assert np.allclose(index.runtime().score(query), [1.0, 2.0, 1.0])

    index.publish(staged.version)
    assert index.active_version == 2
    assert np.allclose(index.runtime().score(query), [1.0, 2.0, 1.6])
    assert version_one_metadata == (
        root / "versions" / "version-00000001" / "version.json"
    ).read_bytes()

    index.rollback(1)
    assert index.active_version == 1
    assert np.allclose(index.runtime().score(query), [1.0, 2.0, 1.0])
    # A reader can pin a generation independently of the active pointer.
    assert np.allclose(
        TieredNumpyRuntime(root, version=2).score(query),
        [1.0, 2.0, 1.6],
    )
    index.rollback(0)
    assert np.allclose(index.runtime().score(query), [1.0, 1.0, 1.0])


def test_failed_or_invalid_updates_do_not_move_active_pointer(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_bank(bank)
    root = tmp_path / "tiered"
    create_versioned_visual_index(
        bank=bank,
        output=root,
        base_route="text",
        visual_route="image",
    )
    index = VersionedVisualIndex(root)
    index.materialize(["item-b"])
    active_before = (root / "active.json").read_bytes()

    with pytest.raises(ValueError, match="unknown item"):
        index.materialize(["missing"])
    with pytest.raises(FileNotFoundError):
        index.rollback(999)

    assert index.active_version == 1
    assert (root / "active.json").read_bytes() == active_before
    pointer = json.loads(active_before)
    assert pointer["version"] == 1


def test_materialization_rejects_a_changed_source_bank(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _write_bank(bank)
    root = tmp_path / "tiered"
    create_versioned_visual_index(
        bank=bank,
        output=root,
        base_route="text",
        visual_route="image",
    )
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected_mutation"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    index = VersionedVisualIndex(root)
    with pytest.raises(ValueError, match="manifest changed"):
        index.materialize(["item-a"])
    assert index.active_version == 0
    assert list((root / "versions").iterdir()) == []


def test_create_rejects_invalid_routes_and_existing_output(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_bank(bank)
    with pytest.raises(ValueError, match="must differ"):
        create_versioned_visual_index(
            bank=bank,
            output=tmp_path / "same",
            base_route="text",
            visual_route="text",
        )
    with pytest.raises(ValueError, match="missing routes"):
        create_versioned_visual_index(
            bank=bank,
            output=tmp_path / "missing",
            base_route="text",
            visual_route="unknown",
        )

    root = tmp_path / "valid"
    create_versioned_visual_index(
        bank=bank,
        output=root,
        base_route="text",
        visual_route="image",
    )
    with pytest.raises(FileExistsError):
        create_versioned_visual_index(
            bank=bank,
            output=root,
            base_route="text",
            visual_route="image",
        )
