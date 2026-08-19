"""Serializable contracts for backbone admission and index compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BackboneProfile:
    """Capabilities required to shorten a document representation lifecycle."""

    name: str
    total_layers: int
    split_after_layer: int
    full_visual_tokens: int
    compact_visual_tokens: int
    exposes_hidden_boundary: bool = True
    exposes_visual_topology: bool = True
    query_independent_documents: bool = True

    @property
    def persistent_fraction(self) -> float:
        return self.compact_visual_tokens / self.full_visual_tokens

    def admission_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.name:
            failures.append("the backbone name is empty")
        if self.total_layers < 2:
            failures.append("the backbone has no separable prefix and suffix")
        if not 0 < self.split_after_layer < self.total_layers:
            failures.append("the split must leave both prefix and suffix layers")
        if not 0 < self.compact_visual_tokens < self.full_visual_tokens:
            failures.append("compact visual capacity must be smaller than Full")
        if not self.exposes_hidden_boundary:
            failures.append("no stable hidden-state boundary is exposed")
        if not self.exposes_visual_topology:
            failures.append("visual token topology is unavailable")
        if not self.query_independent_documents:
            failures.append("document encoding depends on a live query")
        return tuple(failures)

    def validate(self) -> None:
        failures = self.admission_failures()
        if failures:
            raise ValueError("; ".join(failures))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BackboneProfile:
        return cls(**value)


@dataclass(frozen=True)
class CompilePlan:
    """A reproducible physical plan for one model/index configuration."""

    profile: BackboneProfile
    grid_shape: tuple[int, int]
    anchor_layout: str = "right_column_pair_per_2x2"
    assignment: str = "global_hidden_cosine"
    pooling: str = "cluster_mean"
    format_version: int = 1

    def validate(self) -> None:
        self.profile.validate()
        rows, columns = self.grid_shape
        if rows <= 0 or columns <= 0 or rows % 2 or columns % 2:
            raise ValueError("visual grid dimensions must be positive and even")
        if rows * columns != self.profile.full_visual_tokens:
            raise ValueError("visual grid does not match the Full token capacity")
        if self.profile.compact_visual_tokens * 2 != self.profile.full_visual_tokens:
            raise ValueError(
                "the topology-anchored plan currently requires 50% capacity"
            )
        if self.anchor_layout != "right_column_pair_per_2x2":
            raise ValueError("unsupported anchor layout")
        if self.assignment != "global_hidden_cosine":
            raise ValueError("unsupported assignment rule")
        if self.pooling != "cluster_mean":
            raise ValueError("unsupported pooling rule")
        if self.format_version != 1:
            raise ValueError("unsupported compile-plan format")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "grid_shape": list(self.grid_shape),
            "anchor_layout": self.anchor_layout,
            "assignment": self.assignment,
            "pooling": self.pooling,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompilePlan:
        return cls(
            profile=BackboneProfile.from_dict(value["profile"]),
            grid_shape=tuple(value["grid_shape"]),
            anchor_layout=str(value["anchor_layout"]),
            assignment=str(value["assignment"]),
            pooling=str(value["pooling"]),
            format_version=int(value["format_version"]),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()
