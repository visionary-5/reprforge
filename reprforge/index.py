"""Reference late-interaction index, MaxSim scoring and checksummed storage.

The index is a small in-memory reference implementation: it exists so that the
planner, the publication path and the adapter contract can be exercised end to
end without a GPU. Production deployments plug in their own ANN engine and keep
the same manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .versions import VersionManifest

FloatMatrix = NDArray[np.float64]


def normalize_rows(value: ArrayLike) -> FloatMatrix:
    """Return a finite rank-2 matrix with unit-length rows."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("expected a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def maxsim_score(query: ArrayLike, document: ArrayLike) -> float:
    """Score one document with ColBERT-style MaxSim."""

    query_matrix = normalize_rows(query)
    document_matrix = normalize_rows(document)
    if query_matrix.shape[1] != document_matrix.shape[1]:
        raise ValueError("query and document dimensions differ")
    return float(np.max(query_matrix @ document_matrix.T, axis=1).sum())


def target_agreement(
    reference: Iterable[str], candidate: Iterable[str], *, k: int
) -> float:
    """Top-``k`` agreement between a candidate ranking and the reference ranking.

    Averaged over queries this is the paper's Target Agreement (TA@k): the share
    of the exact target index's top-``k`` that a route reproduces.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    reference_top = list(reference)[:k]
    candidate_top = set(list(candidate)[:k])
    if not reference_top:
        raise ValueError("reference ranking is empty")
    return len(candidate_top.intersection(reference_top)) / k


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float


class LateInteractionIndex:
    """In-memory multi-vector index over unit-normalised terminal vectors."""

    def __init__(self, items: Iterable[tuple[str, ArrayLike]]) -> None:
        records = [
            (str(item_id), normalize_rows(vectors)) for item_id, vectors in items
        ]
        if not records:
            raise ValueError("an index requires at least one item")
        identifiers = [item_id for item_id, _ in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("item identifiers must be unique")
        dimensions = {vectors.shape[1] for _, vectors in records}
        if len(dimensions) != 1:
            raise ValueError("all indexed vectors must share one dimension")
        self._items = tuple(records)
        self._by_id = dict(records)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item_id for item_id, _ in self._items)

    @property
    def dimension(self) -> int:
        return int(self._items[0][1].shape[1])

    @property
    def vector_count(self) -> int:
        return sum(len(vectors) for _, vectors in self._items)

    def records(self) -> tuple[tuple[str, FloatMatrix], ...]:
        """Return defensive copies for durable storage backends."""

        return tuple((item_id, vectors.copy()) for item_id, vectors in self._items)

    def contains(self, item_id: str) -> bool:
        return item_id in self._by_id

    def vectors(self, item_id: str) -> FloatMatrix:
        return self._by_id[item_id].copy()

    def search(self, query: ArrayLike, *, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ranking = [
            SearchResult(item_id, maxsim_score(query, document))
            for item_id, document in self._items
        ]
        ranking.sort(key=lambda row: (-row.score, row.item_id))
        return ranking[: min(top_k, len(ranking))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    """Identity and physical shape of one stored index artifact."""

    version: VersionManifest
    source: str
    item_count: int
    vector_count: int
    dimension: int
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "version": self.version.to_dict(),
            "source": self.source,
            "item_count": self.item_count,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexManifest:
        if int(value.get("format_version", 0)) != 3:
            raise ValueError("unsupported index-artifact format")
        source = str(value["source"])
        if not source:
            raise ValueError("index manifest must record its rebuild source")
        return cls(
            version=VersionManifest.from_dict(value["version"]),
            source=source,
            item_count=int(value["item_count"]),
            vector_count=int(value["vector_count"]),
            dimension=int(value["dimension"]),
            payload_sha256=str(value["payload_sha256"]),
        )


def save_index(
    path: str | Path,
    index: LateInteractionIndex,
    version: VersionManifest,
    *,
    source: str = "raw",
) -> IndexManifest:
    """Persist an index, its version and the cut it was built from.

    ``source`` names the rebuild source (``"raw"`` or a cut name) so that a
    generation records whether it was constructed or replayed. Existing paths
    are never overwritten.
    """

    version.validate()
    if not source:
        raise ValueError("source must name the rebuild source")
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
        version=version,
        source=source,
        item_count=len(index),
        vector_count=index.vector_count,
        dimension=index.dimension,
        payload_sha256=_sha256(payload),
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_index(path: str | Path) -> tuple[LateInteractionIndex, IndexManifest]:
    """Load an index only after validating its manifest and payload identity."""

    root = Path(path)
    manifest = IndexManifest.from_dict(json.loads((root / "manifest.json").read_text()))
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
        (str(item_id), vectors[int(offsets[i]) : int(offsets[i + 1])])
        for i, item_id in enumerate(item_ids)
    ]
    loaded = LateInteractionIndex(records)
    observed = (len(loaded), loaded.vector_count, loaded.dimension)
    expected = (manifest.item_count, manifest.vector_count, manifest.dimension)
    if observed != expected:
        raise ValueError("index payload shape does not match its manifest")
    return loaded, manifest
