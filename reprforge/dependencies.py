"""Classify the tensors of an adapter checkpoint by the encoder stage they touch.

The classifier reads only tensor names, never payloads, so it can run on a
safetensors header. It is deliberately conservative: a name that matches no
known stage marks every model component as changed, which forces a raw
rebuild until the classifier is extended for that model family.

Stage vocabulary follows the public ColPali-family checkpoints (Qwen2-VL and
Qwen2.5-VL ``visual.*`` towers and ``visual.merger``; PaliGemma
``vision_tower``/``multi_modal_projector``; Idefics3/SmolVLM ``vision_model``
and ``connector``; ``text_model``/``language_model``/``model.layers`` decoders;
``custom_text_proj`` and similar retrieval heads).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .planning import UpdateScenario

MODEL_COMPONENTS = frozenset({"vision", "base_embedding", "adapter", "projection"})

_VISION_MARKERS = ("visual.", "vision_tower", "vision_model", "vision_encoder")
_MERGER_MARKERS = ("merger", "multi_modal_projector", "mm_projector", "connector")
_PROJECTION_MARKERS = (
    "custom_text_proj",
    "retrieval_projection",
    "1_dense",
    "multi_vector_projector",
    "single_vector_projector",
)
_EMBEDDING_MARKERS = ("embed_tokens", "input_embedding", "wte.")
_DECODER_MARKERS = ("language_model", "text_model", ".model.layers.", "model.layers.")


@dataclass(frozen=True)
class AdapterDependencyScope:
    """Tensor counts per encoder stage for one adapter checkpoint."""

    total_tensors: int
    vision_tensors: int
    merger_tensors: int
    base_embedding_tensors: int
    decoder_tensors: int
    projection_tensors: int
    unknown_tensors: int

    @property
    def changed_components(self) -> frozenset[str]:
        """Version components that this checkpoint changes.

        The merger belongs to the visual prefix: a post-vision state is taken
        after it, so a merger change invalidates that state exactly as a vision
        tower change does.
        """

        components: set[str] = set()
        if self.vision_tensors or self.merger_tensors:
            components.add("vision")
        if self.base_embedding_tensors:
            components.add("base_embedding")
        if self.decoder_tensors:
            components.add("adapter")
        if self.projection_tensors:
            components.add("projection")
        if self.unknown_tensors:
            components.update(MODEL_COMPONENTS)
        return frozenset(components)

    @property
    def post_vision_cut_legal(self) -> bool:
        """Whether a post-vision cut survives this tensor update."""

        return not self.post_vision_cut_blockers

    @property
    def post_vision_cut_blockers(self) -> tuple[str, ...]:
        """Human-readable reasons that force a raw rebuild."""

        blockers: list[str] = []
        if self.vision_tensors:
            blockers.append(f"{self.vision_tensors} tensors update the vision tower")
        if self.merger_tensors:
            blockers.append(f"{self.merger_tensors} tensors update the merger")
        if self.base_embedding_tensors:
            blockers.append(
                f"{self.base_embedding_tensors} tensors update base embeddings"
            )
        if self.unknown_tensors:
            blockers.append(
                f"{self.unknown_tensors} tensors have an unknown dependency"
            )
        return tuple(blockers)

    def to_update_scenario(
        self,
        name: str,
        *,
        expected_count: float = 1.0,
        validation_seconds: float = 0.0,
    ) -> UpdateScenario:
        """Lower the inspected scope into the planner."""

        scenario = UpdateScenario(
            name=name,
            changed_components=self.changed_components,
            expected_count=expected_count,
            validation_seconds=validation_seconds,
        )
        scenario.validate()
        return scenario


def classify_tensor(key: str) -> str:
    """Return the stage of one tensor name: vision, merger, projection,
    base_embedding, decoder or unknown."""

    lowered = key.lower()
    if any(marker in lowered for marker in _MERGER_MARKERS):
        return "merger"
    if any(marker in lowered for marker in _VISION_MARKERS):
        return "vision"
    if any(marker in lowered for marker in _PROJECTION_MARKERS):
        return "projection"
    if any(marker in lowered for marker in _EMBEDDING_MARKERS):
        return "base_embedding"
    if any(marker in lowered for marker in _DECODER_MARKERS):
        return "decoder"
    return "unknown"


def inspect_adapter_tensor_keys(keys: Iterable[str]) -> AdapterDependencyScope:
    """Classify adapter tensors without loading their numerical payloads."""

    counts = {
        "vision": 0,
        "merger": 0,
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
        counts[classify_tensor(key)] += 1
    if observed == 0:
        raise ValueError("an adapter checkpoint must contain at least one tensor")
    return AdapterDependencyScope(
        total_tensors=observed,
        vision_tensors=counts["vision"],
        merger_tensors=counts["merger"],
        base_embedding_tensors=counts["base_embedding"],
        decoder_tensors=counts["decoder"],
        projection_tensors=counts["projection"],
        unknown_tensors=counts["unknown"],
    )
