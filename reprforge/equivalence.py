"""Collection-scoped equivalence certificates for version transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _require_fingerprint(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


@dataclass(frozen=True)
class ComponentOutputFingerprint:
    """Compact digest of one component's ordered outputs on a collection."""

    scope_fingerprint: str
    compared_items: int
    output_fields: tuple[str, ...]
    output_sha256: str

    def validate(self) -> None:
        _require_fingerprint(self.scope_fingerprint, "scope fingerprint")
        if self.compared_items <= 0:
            raise ValueError("output fingerprint must cover at least one item")
        if self.output_fields != tuple(sorted(set(self.output_fields))):
            raise ValueError("output fields must be non-empty, unique, and sorted")
        if not self.output_fields:
            raise ValueError("output fields must be non-empty, unique, and sorted")
        if len(self.output_sha256) != 64:
            raise ValueError("output digest must be a SHA-256 hex string")
        try:
            bytes.fromhex(self.output_sha256)
        except ValueError as error:
            raise ValueError("output digest must be a SHA-256 hex string") from error

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["output_fields"] = list(self.output_fields)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentOutputFingerprint:
        expected = {
            "scope_fingerprint",
            "compared_items",
            "output_fields",
            "output_sha256",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown or missing:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("invalid output fingerprint: " + "; ".join(details))
        fingerprint = cls(
            scope_fingerprint=value["scope_fingerprint"],
            compared_items=value["compared_items"],
            output_fields=tuple(value["output_fields"]),
            output_sha256=value["output_sha256"],
        )
        fingerprint.validate()
        return fingerprint


@dataclass(frozen=True)
class ComponentEquivalence:
    """Exact output equivalence for one component on one collection scope.

    This certificate does not claim that two implementations are universally
    equivalent. It only discharges an otherwise conservative invalidation for
    the exact source/target fingerprints and collection scope recorded here.
    """

    component: str
    source_fingerprint: str
    target_fingerprint: str
    scope_fingerprint: str
    compared_items: int
    output_fields: tuple[str, ...]
    output_sha256: str

    def validate(self) -> None:
        for value, label in (
            (self.component, "component"),
            (self.source_fingerprint, "source fingerprint"),
            (self.target_fingerprint, "target fingerprint"),
            (self.scope_fingerprint, "scope fingerprint"),
        ):
            _require_fingerprint(value, label)
        if self.source_fingerprint == self.target_fingerprint:
            raise ValueError("equivalence is unnecessary for identical fingerprints")
        if self.compared_items <= 0:
            raise ValueError("equivalence must compare at least one item")
        if not self.output_fields:
            raise ValueError("equivalence must declare at least one output field")
        if self.output_fields != tuple(sorted(set(self.output_fields))):
            raise ValueError("output fields must be unique and sorted")
        if len(self.output_sha256) != 64:
            raise ValueError("output digest must be a SHA-256 hex string")
        try:
            bytes.fromhex(self.output_sha256)
        except ValueError as error:
            raise ValueError("output digest must be a SHA-256 hex string") from error

    def covers(
        self,
        component: str,
        source_fingerprint: str,
        target_fingerprint: str,
        scope_fingerprint: str,
    ) -> bool:
        """Return whether this certificate covers an exact transition."""

        self.validate()
        return (
            self.component == component
            and self.source_fingerprint == source_fingerprint
            and self.target_fingerprint == target_fingerprint
            and self.scope_fingerprint == scope_fingerprint
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["output_fields"] = list(self.output_fields)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentEquivalence:
        expected = {
            "component",
            "source_fingerprint",
            "target_fingerprint",
            "scope_fingerprint",
            "compared_items",
            "output_fields",
            "output_sha256",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown or missing:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("invalid equivalence certificate: " + "; ".join(details))
        certificate = cls(
            component=value["component"],
            source_fingerprint=value["source_fingerprint"],
            target_fingerprint=value["target_fingerprint"],
            scope_fingerprint=value["scope_fingerprint"],
            compared_items=value["compared_items"],
            output_fields=tuple(value["output_fields"]),
            output_sha256=value["output_sha256"],
        )
        certificate.validate()
        return certificate


def _update_digest(
    digest: Any,
    item_index: int,
    field: str,
    array: np.ndarray[Any, Any],
) -> None:
    metadata = json.dumps(
        {
            "item": item_index,
            "field": field,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload = np.ascontiguousarray(array).tobytes()
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def fingerprint_component_outputs(
    *,
    scope_fingerprint: str,
    outputs: Iterable[Mapping[str, Any]],
) -> ComponentOutputFingerprint:
    """Digest an ordered stream of NumPy-compatible component outputs."""

    _require_fingerprint(scope_fingerprint, "scope fingerprint")
    digest = hashlib.sha256()
    expected_fields: tuple[str, ...] | None = None
    compared_items = 0
    for item_index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            raise TypeError("component outputs must be mappings")
        fields = tuple(sorted(output))
        if not fields:
            raise ValueError(f"output fields are empty at item {item_index}")
        if expected_fields is None:
            expected_fields = fields
        elif fields != expected_fields:
            raise ValueError(f"output contract changed at item {item_index}")
        for field in fields:
            array = np.asarray(output[field])
            if array.dtype.hasobject:
                raise ValueError("object arrays cannot be output-fingerprinted")
            _update_digest(digest, item_index, field, array)
        compared_items += 1
    fingerprint = ComponentOutputFingerprint(
        scope_fingerprint=scope_fingerprint,
        compared_items=compared_items,
        output_fields=expected_fields or (),
        output_sha256=digest.hexdigest(),
    )
    fingerprint.validate()
    return fingerprint


def certify_component_fingerprints(
    *,
    component: str,
    source_component_fingerprint: str,
    target_component_fingerprint: str,
    source_outputs: ComponentOutputFingerprint,
    target_outputs: ComponentOutputFingerprint,
) -> ComponentEquivalence:
    """Certify two persisted collection-output fingerprints as equivalent."""

    source_outputs.validate()
    target_outputs.validate()
    if source_outputs.scope_fingerprint != target_outputs.scope_fingerprint:
        raise ValueError("source and target output scopes differ")
    if (
        source_outputs.compared_items != target_outputs.compared_items
        or source_outputs.output_fields != target_outputs.output_fields
        or source_outputs.output_sha256 != target_outputs.output_sha256
    ):
        raise ValueError("component outputs differ on the collection scope")
    certificate = ComponentEquivalence(
        component=component,
        source_fingerprint=source_component_fingerprint,
        target_fingerprint=target_component_fingerprint,
        scope_fingerprint=source_outputs.scope_fingerprint,
        compared_items=source_outputs.compared_items,
        output_fields=source_outputs.output_fields,
        output_sha256=source_outputs.output_sha256,
    )
    certificate.validate()
    return certificate


def certify_component_equivalence(
    *,
    component: str,
    source_fingerprint: str,
    target_fingerprint: str,
    scope_fingerprint: str,
    source_outputs: Iterable[Mapping[str, Any]],
    target_outputs: Iterable[Mapping[str, Any]],
) -> ComponentEquivalence:
    """Compare two output streams exactly and return an auditable certificate.

    Each iterable item is one collection record and maps stable output names to
    NumPy-compatible tensors. Shapes and dtypes are part of the contract.
    Object arrays are rejected because their byte representation is not a
    portable computation fingerprint.
    """

    for value, label in (
        (component, "component"),
        (source_fingerprint, "source fingerprint"),
        (target_fingerprint, "target fingerprint"),
        (scope_fingerprint, "scope fingerprint"),
    ):
        _require_fingerprint(value, label)
    if source_fingerprint == target_fingerprint:
        raise ValueError("equivalence is unnecessary for identical fingerprints")

    source_output_fingerprint = fingerprint_component_outputs(
        scope_fingerprint=scope_fingerprint,
        outputs=source_outputs,
    )
    target_output_fingerprint = fingerprint_component_outputs(
        scope_fingerprint=scope_fingerprint,
        outputs=target_outputs,
    )
    return certify_component_fingerprints(
        component=component,
        source_component_fingerprint=source_fingerprint,
        target_component_fingerprint=target_fingerprint,
        source_outputs=source_output_fingerprint,
        target_outputs=target_output_fingerprint,
    )
