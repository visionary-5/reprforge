import json
from pathlib import Path

import pytest

from reprforge import (
    VersionManifest,
    publish_generation,
    resolve_active_generation,
    seal_generation,
    validate_generation,
)


def version(adapter: str) -> VersionManifest:
    return VersionManifest(
        source="pages-v1",
        processor="processor-v1",
        vision="vision-v1",
        base_embedding="embedding-v1",
        adapter=adapter,
        projection="projection-v1",
        index_policy="sq8-v1",
    )


def generation(root: Path, name: str, payload: bytes) -> Path:
    target = root / "generations" / name
    target.mkdir(parents=True)
    (target / "terminal.bin").write_bytes(payload)
    (target / "candidate.bin").write_bytes(payload[::-1])
    return target


def test_seal_publish_and_switch_immutable_generations(tmp_path: Path) -> None:
    generation(tmp_path, "adapter-v1", b"old-terminal")
    generation(tmp_path, "adapter-v2", b"new-terminal")
    old = seal_generation(
        tmp_path,
        "adapter-v1",
        ("terminal.bin", "candidate.bin"),
        version=version("adapter-v1"),
    )
    new = seal_generation(
        tmp_path,
        "adapter-v2",
        ("terminal.bin", "candidate.bin"),
        version=version("adapter-v2"),
    )

    assert publish_generation(tmp_path, "adapter-v1") == old
    active_path, active = resolve_active_generation(tmp_path)
    assert active_path.name == "adapter-v1"
    assert active == old

    assert publish_generation(tmp_path, "adapter-v2") == new
    active_path, active = resolve_active_generation(tmp_path)
    assert active_path.name == "adapter-v2"
    assert active == new


def test_generation_validation_rejects_mutated_payload(tmp_path: Path) -> None:
    target = generation(tmp_path, "v1", b"original")
    seal_generation(tmp_path, "v1", ("terminal.bin", "candidate.bin"))
    (target / "terminal.bin").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="checksum|size"):
        validate_generation(tmp_path, "v1")


def test_generation_sealing_rejects_escape_and_overwrite(tmp_path: Path) -> None:
    target = generation(tmp_path, "v1", b"payload")

    with pytest.raises(ValueError, match="stay inside"):
        seal_generation(tmp_path, "v1", ("../outside",))

    (target / "linked.bin").symlink_to(target / "terminal.bin")
    with pytest.raises(ValueError, match="symlinks"):
        seal_generation(tmp_path, "v1", ("linked.bin",))

    seal_generation(tmp_path, "v1", ("terminal.bin",))
    with pytest.raises(FileExistsError):
        seal_generation(tmp_path, "v1", ("terminal.bin",))


def test_active_pointer_is_bound_to_manifest_hash(tmp_path: Path) -> None:
    target = generation(tmp_path, "v1", b"payload")
    seal_generation(tmp_path, "v1", ("terminal.bin",))
    publish_generation(tmp_path, "v1")
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    (target / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest checksum"):
        resolve_active_generation(tmp_path)
