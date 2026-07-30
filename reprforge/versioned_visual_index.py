#!/usr/bin/env python3
"""Minimal two-tier index with immutable visual delta generations.

The base tier stores one cheap representation for every item.  A published
delta generation stores the expensive visual representation only for items
that have been materialized so far.  Query execution scores the base, scores
the active delta, and *replaces* base scores for delta-resident items.

Generations are immutable.  Publishing or rolling back changes only one small
atomic pointer, so a reader observes either the old generation or the new one.
The first implementation intentionally assumes one writer; multi-process
locking, eviction, and background queues belong to later system layers.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reprforge.heterogeneous_index import (
    FORMAT_VERSION,
    NumpyMaxSimRuntime,
    _json,
    _load_shard,
    _read_manifest,
    _sha256,
    _write_shard,
    compile_heterogeneous_index,
)


TIERED_FORMAT = "reprforge-versioned-visual-index"
ACTIVE_FORMAT = "reprforge-active-visual-generation"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON pointer atomically within one filesystem directory."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _root_manifest(root: Path) -> dict[str, Any]:
    return _read_manifest(root, TIERED_FORMAT)


def _active_version(root: Path) -> int:
    payload = json.loads((root / "active.json").read_text(encoding="utf-8"))
    if payload.get("format") != ACTIVE_FORMAT:
        raise ValueError(f"invalid active pointer under {root}")
    version = int(payload["version"])
    if version < 0:
        raise ValueError("active version cannot be negative")
    return version


def _version_name(version: int) -> str:
    if version <= 0:
        raise ValueError("materialized versions must be positive")
    return f"version-{version:08d}"


def _version_path(root: Path, version: int) -> Path:
    return root / "versions" / _version_name(version)


def _version_payload(root: Path, version: int) -> dict[str, Any]:
    if version == 0:
        return {
            "version": 0,
            "parent_version": None,
            "item_ids": [],
            "new_item_ids": [],
        }
    path = _version_path(root, version) / "version.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != version:
        raise ValueError(f"version metadata mismatch under {path.parent}")
    return payload


def create_versioned_visual_index(
    *,
    bank: Path,
    output: Path,
    base_route: str,
    visual_route: str,
    storage_dtype: str | None = None,
) -> dict[str, Any]:
    """Create an all-base index with an empty visual generation."""

    bank_manifest = _read_manifest(bank, "reprforge-embedding-bank")
    routes = set(bank_manifest["routes"])
    missing = {base_route, visual_route} - routes
    if missing:
        raise ValueError(f"embedding bank is missing routes: {sorted(missing)}")
    if base_route == visual_route:
        raise ValueError("base and visual routes must differ")
    if output.exists():
        raise FileExistsError(f"versioned index already exists: {output}")

    base_ids, base_vectors, _ = _load_shard(bank / "routes" / base_route)
    visual_ids, visual_vectors, _ = _load_shard(bank / "routes" / visual_route)
    if base_ids != visual_ids:
        raise ValueError("base and visual routes do not share one item order")
    if base_vectors.shape[1] != visual_vectors.shape[1]:
        raise ValueError("base and visual routes have different dimensions")

    output.mkdir(parents=True)
    (output / "versions").mkdir()
    base_plan = {item_id: base_route for item_id in base_ids}
    base_manifest = compile_heterogeneous_index(
        bank=bank,
        plan=base_plan,
        output=output / "base",
        storage_dtype=storage_dtype,
    )
    manifest = {
        "format": TIERED_FORMAT,
        "format_version": FORMAT_VERSION,
        "source_bank": str(bank.resolve()),
        "source_bank_manifest_sha256": _sha256(bank / "manifest.json"),
        "base_route": base_route,
        "visual_route": visual_route,
        "storage_dtype": str(base_manifest["storage_dtype"]),
        "dimension": int(base_manifest["dimension"]),
        "item_count": len(base_ids),
        "base_compact_vector_bytes": int(base_manifest["compact_vector_bytes"]),
        "writer_contract": "single-writer-atomic-publish",
        "merge_semantics": "active-delta-overrides-base",
    }
    _json(output / "manifest.json", manifest)
    _atomic_json(
        output / "active.json",
        {
            "format": ACTIVE_FORMAT,
            "format_version": FORMAT_VERSION,
            "version": 0,
        },
    )
    return manifest


@dataclass(frozen=True)
class StagedGeneration:
    version: int
    parent_version: int
    item_ids: tuple[str, ...]
    new_item_ids: tuple[str, ...]
    compact_vector_bytes: int


class VersionedVisualIndex:
    """Single-writer control plane for an immutable base + visual delta."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = _root_manifest(root)
        self.bank = Path(str(self.manifest["source_bank"]))
        self.base_runtime = NumpyMaxSimRuntime(root / "base")
        self._positions = {
            item_id: index
            for index, item_id in enumerate(self.base_runtime.item_ids)
        }

    @property
    def active_version(self) -> int:
        return _active_version(self.root)

    @property
    def cached_item_ids(self) -> frozenset[str]:
        return frozenset(
            str(item_id)
            for item_id in _version_payload(
                self.root, self.active_version
            )["item_ids"]
        )

    def _next_version(self) -> int:
        versions = [
            int(path.name.removeprefix("version-"))
            for path in (self.root / "versions").iterdir()
            if path.is_dir() and path.name.startswith("version-")
        ]
        return max(versions, default=0) + 1

    def _validate_source_bank(self) -> None:
        observed = _sha256(self.bank / "manifest.json")
        expected = str(self.manifest["source_bank_manifest_sha256"])
        if observed != expected:
            raise ValueError(
                "source embedding bank manifest changed after index creation"
            )

    def stage(self, item_ids: Iterable[str]) -> StagedGeneration | None:
        """Build, but do not publish, the union of active and requested items."""

        requested = tuple(sorted(set(str(item_id) for item_id in item_ids)))
        self._validate_source_bank()
        unknown = sorted(set(requested) - set(self._positions))
        if unknown:
            raise ValueError(f"unknown item identifiers: {unknown[:5]}")
        parent = self.active_version
        existing = self.cached_item_ids
        new_items = tuple(item_id for item_id in requested if item_id not in existing)
        if not new_items:
            return None
        selected = tuple(
            item_id
            for item_id in self.base_runtime.item_ids
            if item_id in existing or item_id in set(new_items)
        )

        version = self._next_version()
        output = _version_path(self.root, version)
        output.mkdir(parents=True)
        route = str(self.manifest["visual_route"])
        source_ids, vectors, offsets = _load_shard(
            self.bank / "routes" / route
        )
        source_positions = {
            item_id: index for index, item_id in enumerate(source_ids)
        }
        embeddings = [
            np.asarray(
                vectors[
                    offsets[source_positions[item_id]] :
                    offsets[source_positions[item_id] + 1]
                ]
            )
            for item_id in selected
        ]
        shard_manifest = _write_shard(
            output / "index" / "routes" / route,
            identifiers=selected,
            embeddings=embeddings,
            storage_dtype=str(self.manifest["storage_dtype"]),
        )
        _json(
            output / "index" / "items.json",
            [
                {
                    "item_id": item_id,
                    "route": route,
                    "global_position": position,
                }
                for position, item_id in enumerate(selected)
            ],
        )
        delta_manifest = {
            "format": "reprforge-heterogeneous-index",
            "format_version": FORMAT_VERSION,
            "source_bank": str(self.bank),
            "storage_dtype": str(self.manifest["storage_dtype"]),
            "dimension": int(self.manifest["dimension"]),
            "item_count": len(selected),
            "compact_vector_bytes": int(shard_manifest["vector_bytes"]),
            "storage_padding_bytes": 0,
            "routes": {route: shard_manifest},
            "route_counts": {route: len(selected)},
        }
        _json(output / "index" / "manifest.json", delta_manifest)
        version_payload = {
            "format": "reprforge-visual-generation",
            "format_version": FORMAT_VERSION,
            "version": version,
            "parent_version": parent,
            "item_ids": list(selected),
            "new_item_ids": list(new_items),
            "compact_vector_bytes": int(shard_manifest["vector_bytes"]),
        }
        _json(output / "version.json", version_payload)
        return StagedGeneration(
            version=version,
            parent_version=parent,
            item_ids=selected,
            new_item_ids=new_items,
            compact_vector_bytes=int(shard_manifest["vector_bytes"]),
        )

    def publish(self, version: int) -> None:
        if version != 0:
            _version_payload(self.root, version)
        _atomic_json(
            self.root / "active.json",
            {
                "format": ACTIVE_FORMAT,
                "format_version": FORMAT_VERSION,
                "version": version,
            },
        )

    def materialize(self, item_ids: Iterable[str]) -> dict[str, Any]:
        """Stage and atomically publish new visual items; cache hits are no-ops."""

        requested = tuple(sorted(set(str(item_id) for item_id in item_ids)))
        staged = self.stage(requested)
        if staged is None:
            return {
                "version": self.active_version,
                "published": False,
                "requested_items": len(requested),
                "new_items": 0,
                "cache_hits": len(requested),
            }
        self.publish(staged.version)
        return {
            "version": staged.version,
            "parent_version": staged.parent_version,
            "published": True,
            "requested_items": len(requested),
            "new_items": len(staged.new_item_ids),
            "cache_hits": len(requested) - len(staged.new_item_ids),
            "cached_items": len(staged.item_ids),
            "compact_vector_bytes": staged.compact_vector_bytes,
        }

    def rollback(self, version: int) -> None:
        """Atomically point readers at a prior immutable generation."""

        self.publish(version)

    def status(self) -> dict[str, Any]:
        version = self.active_version
        payload = _version_payload(self.root, version)
        generations = sorted(
            int(path.name.removeprefix("version-"))
            for path in (self.root / "versions").iterdir()
            if (
                path.is_dir()
                and path.name.startswith("version-")
                and (path / "version.json").is_file()
            )
        )
        return {
            "active_version": version,
            "available_versions": [0, *generations],
            "cached_items": len(payload["item_ids"]),
            "base_items": len(self.base_runtime.item_ids),
            "base_compact_vector_bytes": self.base_runtime.compact_vector_bytes,
            "delta_compact_vector_bytes": int(
                payload.get("compact_vector_bytes", 0)
            ),
            "merge_semantics": str(self.manifest["merge_semantics"]),
        }

    def runtime(self, *, version: int | None = None) -> "TieredNumpyRuntime":
        return TieredNumpyRuntime(
            self.root,
            version=self.active_version if version is None else version,
        )


class TieredNumpyRuntime:
    """Exact reference query path for base plus one visual generation."""

    def __init__(self, root: Path, *, version: int | None = None) -> None:
        self.root = root
        self.manifest = _root_manifest(root)
        self.base = NumpyMaxSimRuntime(root / "base")
        self.item_ids = self.base.item_ids
        self._positions = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        self.version = _active_version(root) if version is None else version
        if self.version == 0:
            self.delta = None
        else:
            _version_payload(root, self.version)
            self.delta = NumpyMaxSimRuntime(
                _version_path(root, self.version) / "index"
            )
            unknown = set(self.delta.item_ids) - set(self._positions)
            if unknown:
                raise ValueError(
                    f"delta contains unknown base items: {sorted(unknown)[:5]}"
                )

    @property
    def cached_item_ids(self) -> frozenset[str]:
        return (
            frozenset()
            if self.delta is None
            else frozenset(self.delta.item_ids)
        )

    @property
    def compact_vector_bytes(self) -> int:
        delta_bytes = 0 if self.delta is None else self.delta.compact_vector_bytes
        return self.base.compact_vector_bytes + delta_bytes

    def score(self, query_embedding: Any) -> np.ndarray:
        scores = self.base.score(query_embedding)
        if self.delta is None:
            return scores
        delta_scores = self.delta.score(query_embedding)
        for item_id, score in zip(
            self.delta.item_ids, delta_scores, strict=True
        ):
            scores[self._positions[item_id]] = score
        return scores

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.score(query_embedding)
        ranked = sorted(
            range(len(self.item_ids)),
            key=lambda index: (-float(scores[index]), self.item_ids[index]),
        )
        return [
            (self.item_ids[index], float(scores[index]))
            for index in ranked[:top_k]
        ]


def _read_item_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("item_ids")
    if not isinstance(payload, list):
        raise ValueError("item file must be a JSON list or {'item_ids': [...]}")
    return [str(item_id) for item_id in payload]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--bank", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--base-route", required=True)
    create.add_argument("--visual-route", required=True)
    create.add_argument("--storage-dtype", choices=("float16", "float32"))

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--index", type=Path, required=True)
    materialize.add_argument("--item-id", action="append", default=[])
    materialize.add_argument("--items-file", type=Path)

    rollback = commands.add_parser("rollback")
    rollback.add_argument("--index", type=Path, required=True)
    rollback.add_argument("--version", type=int, required=True)

    status = commands.add_parser("status")
    status.add_argument("--index", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "create":
        result = create_versioned_visual_index(
            bank=args.bank,
            output=args.output,
            base_route=args.base_route,
            visual_route=args.visual_route,
            storage_dtype=args.storage_dtype,
        )
    elif args.command == "materialize":
        item_ids = list(args.item_id)
        if args.items_file is not None:
            item_ids.extend(_read_item_ids(args.items_file))
        if not item_ids:
            parser.error("materialize requires --item-id or --items-file")
        result = VersionedVisualIndex(args.index).materialize(item_ids)
    elif args.command == "rollback":
        index = VersionedVisualIndex(args.index)
        index.rollback(args.version)
        result = index.status()
    else:
        result = VersionedVisualIndex(args.index).status()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
