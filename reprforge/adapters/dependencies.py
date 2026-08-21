"""Conservative dependency admission for versioned model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..planning import UpdateScenario


_ALL_MODEL_COMPONENTS = frozenset(
    {"vision", "base_embedding", "adapter", "projection"}
)


@dataclass(frozen=True)
class AdapterDependencyScope:
    """Tensor-level components changed by one adapter checkpoint.

    ReprForge treats unrecognized tensors conservatively. A new model family
    must extend the classifier before an intermediate artifact can be reused.
    """

    total_tensors: int
    vision_tensors: int
    base_embedding_tensors: int
    decoder_tensors: int
    projection_tensors: int
    unknown_tensors: int

    @property
    def changed_components(self) -> frozenset[str]:
        components: set[str] = set()
        if self.vision_tensors:
            components.add("vision")
        if self.base_embedding_tensors:
            components.add("base_embedding")
        if self.decoder_tensors:
            components.add("adapter")
        if self.projection_tensors:
            components.add("projection")
        if self.unknown_tensors:
            components.update(_ALL_MODEL_COMPONENTS)
        return frozenset(components)

    @property
    def post_vision_replay_valid(self) -> bool:
        """Whether a post-vision artifact survives this tensor update."""

        return not self.post_vision_replay_blockers

    @property
    def post_vision_replay_blockers(self) -> tuple[str, ...]:
        """Human-readable reasons that force a raw-evidence rebuild."""

        blockers: list[str] = []
        if self.vision_tensors:
            blockers.append(f"{self.vision_tensors} adapter tensors update vision")
        if self.base_embedding_tensors:
            blockers.append(
                f"{self.base_embedding_tensors} adapter tensors update base embeddings"
            )
        if self.unknown_tensors:
            blockers.append(
                f"{self.unknown_tensors} adapter tensors have an unknown dependency"
            )
        return tuple(blockers)

    def to_update_scenario(
        self,
        name: str,
        *,
        expected_count: float = 1.0,
    ) -> UpdateScenario:
        """Lower the inspected scope into the materialization planner."""

        scenario = UpdateScenario(
            name=name,
            changed_components=self.changed_components,
            expected_count=expected_count,
        )
        scenario.validate()
        return scenario


def inspect_adapter_tensor_keys(keys: Iterable[str]) -> AdapterDependencyScope:
    """Classify adapter tensors without loading their numerical payloads.

    Classification uses stable module-path concepts rather than one model's
    exact prefix. Vision is checked first because nested paths also contain
    generic attention names. Unknown paths invalidate model-intermediate reuse.
    """

    counts = {
        "vision": 0,
        "base_embedding": 0,
        "decoder": 0,
        "projection": 0,
        "unknown": 0,
    }
    observed = 0
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValueError("adapter tensor keys must be non-empty strings")
        observed += 1
        lowered = key.lower()
        if "vision" in lowered:
            counts["vision"] += 1
        elif "custom_text_proj" in lowered or "retrieval_projection" in lowered:
            counts["projection"] += 1
        elif "embed_tokens" in lowered or "input_embedding" in lowered:
            counts["base_embedding"] += 1
        elif "language_model" in lowered or ".model.layers." in lowered:
            counts["decoder"] += 1
        else:
            counts["unknown"] += 1
    if observed == 0:
        raise ValueError("an adapter checkpoint must contain at least one tensor")
    return AdapterDependencyScope(
        total_tensors=observed,
        vision_tensors=counts["vision"],
        base_embedding_tensors=counts["base_embedding"],
        decoder_tensors=counts["decoder"],
        projection_tensors=counts["projection"],
        unknown_tensors=counts["unknown"],
    )
