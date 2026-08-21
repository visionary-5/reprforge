"""Checksummed immutable generations and atomic active-generation publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..planning import VersionManifest

_GENERATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_generation_name(value: str) -> None:
    if not _GENERATION_NAME.fullmatch(value):
        raise ValueError("generation names may contain only letters, digits, ._-")


def _artifact_path(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or not logical.parts or ".." in logical.parts:
        raise ValueError(f"artifact path must stay inside its generation: {relative}")
    path = root
    for part in logical.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"generation artifacts may not use symlinks: {relative}")
    if not path.is_file():
        raise ValueError(f"artifact must be a regular file: {relative}")
    return path


@dataclass(frozen=True)
class GenerationArtifact:
    """One immutable file sealed into a generation manifest."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class GenerationManifest:
    """Identity of one complete, immutable serving generation."""

    generation: str
    artifacts: tuple[GenerationArtifact, ...]
    version: VersionManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format_version": 1,
            "generation": self.generation,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }
        if self.version is not None:
            value["version"] = self.version.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GenerationManifest:
        if int(value.get("format_version", 0)) != 1:
            raise ValueError("unsupported generation-manifest format")
        generation = str(value["generation"])
        _validate_generation_name(generation)
        artifacts = tuple(
            GenerationArtifact(
                path=str(row["path"]),
                bytes=int(row["bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in value["artifacts"]
        )
        if not artifacts or len({row.path for row in artifacts}) != len(artifacts):
            raise ValueError("a generation needs unique, non-empty artifacts")
        version = (
            VersionManifest.from_dict(value["version"])
            if "version" in value
            else None
        )
        return cls(generation, artifacts, version)


def seal_generation(
    deployment_root: str | Path,
    generation: str,
    artifacts: Iterable[str],
    *,
    version: VersionManifest | None = None,
) -> GenerationManifest:
    """Hash existing generation files and write a new immutable manifest.

    Files must already live under ``generations/<generation>``. Symlinks and
    paths that escape that directory are rejected. The manifest itself is
    create-only: a changed generation receives a new name.
    """

    _validate_generation_name(generation)
    if version is not None:
        version.validate()
    deployment = Path(deployment_root)
    root = deployment / "generations" / generation
    if not root.is_dir() or root.is_symlink():
        raise ValueError("generation directory must exist and may not be a symlink")
    names = tuple(artifacts)
    if not names or len(set(names)) != len(names):
        raise ValueError("artifact paths must be unique and non-empty")
    records = tuple(
        GenerationArtifact(name, path.stat().st_size, _sha256(path))
        for name in sorted(names)
        for path in (_artifact_path(root, name),)
    )
    manifest = GenerationManifest(generation, records, version)
    manifest_path = root / "manifest.json"
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(root)
    return manifest


def validate_generation(
    deployment_root: str | Path,
    generation: str,
) -> GenerationManifest:
    """Validate every file named by an immutable generation manifest."""

    _validate_generation_name(generation)
    root = Path(deployment_root) / "generations" / generation
    if root.is_symlink() or not root.is_dir():
        raise ValueError("generation directory is missing or is a symlink")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("generation manifest is missing or is a symlink")
    manifest = GenerationManifest.from_dict(json.loads(manifest_path.read_text()))
    if manifest.generation != generation:
        raise ValueError("generation name does not match its manifest")
    for artifact in manifest.artifacts:
        path = _artifact_path(root, artifact.path)
        if path.stat().st_size != artifact.bytes:
            raise ValueError(f"generation artifact size changed: {artifact.path}")
        if _sha256(path) != artifact.sha256:
            raise ValueError(f"generation artifact checksum changed: {artifact.path}")
    return manifest


def publish_generation(
    deployment_root: str | Path,
    generation: str,
) -> GenerationManifest:
    """Validate and atomically make one immutable generation active."""

    deployment = Path(deployment_root)
    manifest = validate_generation(deployment, generation)
    manifest_path = deployment / "generations" / generation / "manifest.json"
    pointer = {
        "format_version": 1,
        "generation": generation,
        "manifest_sha256": _sha256(manifest_path),
    }
    temporary = deployment / f".ACTIVE.{uuid.uuid4().hex}.tmp"
    with temporary.open("x") as handle:
        handle.write(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, deployment / "ACTIVE.json")
    _fsync_directory(deployment)
    return manifest


def resolve_active_generation(
    deployment_root: str | Path,
) -> tuple[Path, GenerationManifest]:
    """Resolve one pointer snapshot and validate the referenced generation."""

    deployment = Path(deployment_root)
    pointer = json.loads((deployment / "ACTIVE.json").read_text())
    if int(pointer.get("format_version", 0)) != 1:
        raise ValueError("unsupported active-generation pointer format")
    generation = str(pointer["generation"])
    _validate_generation_name(generation)
    manifest_path = deployment / "generations" / generation / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("active-generation manifest is missing or is a symlink")
    if _sha256(manifest_path) != str(pointer["manifest_sha256"]):
        raise ValueError("active-generation manifest checksum changed")
    return manifest_path.parent, validate_generation(deployment, generation)
