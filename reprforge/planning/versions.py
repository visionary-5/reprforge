"""Version fingerprints for dependency-correct index maintenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .materialization import UpdateScenario


_COMPONENTS = (
    "source",
    "processor",
    "vision",
    "base_embedding",
    "adapter",
    "projection",
    "index_policy",
)


@dataclass(frozen=True)
class VersionManifest:
    """Content or semantic fingerprints that define one index version."""

    source: str
    processor: str
    vision: str
    base_embedding: str
    adapter: str
    projection: str
    index_policy: str

    def validate(self) -> None:
        missing = [
            name
            for name in _COMPONENTS
            if not isinstance(getattr(self, name), str) or not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                "version fingerprints must be non-empty: " + ", ".join(missing)
            )

    def changed_components(self, target: VersionManifest) -> frozenset[str]:
        """Return the exact dependency delta from this version to ``target``."""

        self.validate()
        target.validate()
        return frozenset(
            name for name in _COMPONENTS if getattr(self, name) != getattr(target, name)
        )

    def update_scenario(
        self,
        target: VersionManifest,
        name: str,
        *,
        expected_count: float = 1.0,
    ) -> UpdateScenario:
        """Lower a manifest diff into the materialization planner."""

        changed = self.changed_components(target)
        if not changed:
            raise ValueError("source and target version manifests are identical")
        scenario = UpdateScenario(name, changed, expected_count)
        scenario.validate()
        return scenario

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VersionManifest:
        unknown = sorted(set(value) - set(_COMPONENTS))
        missing = sorted(set(_COMPONENTS) - set(value))
        if unknown or missing:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("invalid version manifest: " + "; ".join(details))
        manifest = cls(**{name: value[name] for name in _COMPONENTS})
        manifest.validate()
        return manifest
