"""Versioned on-disk artifacts for compiled multi-vector indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..planning import CompilePlan, VersionManifest
from .late_interaction import CompactIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    """Identity and physical shape of a compiled index artifact."""

    plan: CompilePlan
    item_count: int
    vector_count: int
    dimension: int
    payload_sha256: str
    version: VersionManifest | None = None

    @property
    def format_version(self) -> int:
        return 2 if self.version is not None else 1

    def to_dict(self) -> dict[str, Any]:
        value = {
            "format_version": self.format_version,
            "plan": self.plan.to_dict(),
            "plan_fingerprint": self.plan.fingerprint,
            "item_count": self.item_count,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "payload_sha256": self.payload_sha256,
        }
        if self.version is not None:
            value["version"] = self.version.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexManifest:
        format_version = int(value["format_version"])
        if format_version not in (1, 2):
            raise ValueError("unsupported index-artifact format")
        if format_version == 1 and "version" in value:
            raise ValueError("format 1 cannot contain a version manifest")
        if format_version == 2 and "version" not in value:
            raise ValueError("format 2 requires a version manifest")
        plan = CompilePlan.from_dict(value["plan"])
        if value["plan_fingerprint"] != plan.fingerprint:
            raise ValueError("compile-plan fingerprint mismatch")
        return cls(
            plan=plan,
            item_count=int(value["item_count"]),
            vector_count=int(value["vector_count"]),
            dimension=int(value["dimension"]),
            payload_sha256=str(value["payload_sha256"]),
            version=(
                VersionManifest.from_dict(value["version"])
                if format_version == 2
                else None
            ),
        )


def save_index(
    path: str | Path,
    index: CompactIndex,
    plan: CompilePlan,
    version: VersionManifest | None = None,
) -> IndexManifest:
    """Persist an index and its exact physical plan without overwriting."""

    if version is not None:
        version.validate()
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    records = index.records()
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for position, (_, vectors) in enumerate(records, start=1):
        offsets[position] = offsets[position - 1] + len(vectors)
    payload = root / "vectors.npz"
    np.savez_compressed(
        payload,
        vectors=np.concatenate([vectors for _, vectors in records]),
        offsets=offsets,
        item_ids=np.asarray([item_id for item_id, _ in records]),
    )
    manifest = IndexManifest(
        plan=plan,
        item_count=len(index),
        vector_count=index.vector_count,
        dimension=index.dimension,
        payload_sha256=_sha256(payload),
        version=version,
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_index(path: str | Path) -> tuple[CompactIndex, IndexManifest]:
    """Load an index only after validating its plan and payload identity."""

    root = Path(path)
    manifest = IndexManifest.from_dict(
        json.loads((root / "manifest.json").read_text())
    )
    payload = root / "vectors.npz"
    if _sha256(payload) != manifest.payload_sha256:
        raise ValueError("index payload checksum mismatch")
    with np.load(payload, allow_pickle=False) as arrays:
        vectors = arrays["vectors"]
        offsets = arrays["offsets"]
        item_ids = arrays["item_ids"]
    if len(offsets) != len(item_ids) + 1:
        raise ValueError("corrupt index offsets")
    records = [
        (
            str(item_id),
            vectors[int(offsets[index]) : int(offsets[index + 1])],
        )
        for index, item_id in enumerate(item_ids)
    ]
    loaded = CompactIndex(records)
    observed = (len(loaded), loaded.vector_count, loaded.dimension)
    expected = (
        manifest.item_count,
        manifest.vector_count,
        manifest.dimension,
    )
    if observed != expected:
        raise ValueError("index payload shape does not match its manifest")
    return loaded, manifest
