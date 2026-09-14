"""Quickstart: an adapter upgrade routed through a semantic recompilation cut.

Runs on CPU with a synthetic two-stage encoder that mimics a ColPali-style
retriever: a frozen visual prefix shared by both versions and a
version-specific suffix. It shows the three things the library does:

1. decide which components a release changes (version tuple + tensor census);
2. pick the rebuild source under storage and quality contracts (planner);
3. replay the target suffix from a stored cut and verify against the exact
   target index with Target Agreement, then seal and publish the generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from reprforge import (
    CutState,
    LateInteractionIndex,
    MaterializationOption,
    VersionManifest,
    choose_materializations,
    cut_leverage,
    inspect_adapter_tensor_keys,
    normalize_rows,
    publish_generation,
    resolve_active_generation,
    save_index,
    seal_generation,
    target_agreement,
)


class SyntheticAdapter:
    """prefix = processor + vision tower + merger (frozen); suffix = LoRA + head."""

    cuts = ("post_vision",)

    def __init__(self, prefix: np.ndarray, suffix: np.ndarray) -> None:
        self.prefix, self.suffix = prefix, suffix

    def encode(self, page: object) -> np.ndarray:
        return normalize_rows(np.tanh(np.asarray(page) @ self.prefix) @ self.suffix)

    def emit_cut(self, page: object, cut: str) -> CutState:
        return CutState(
            cut=cut,
            state=np.tanh(np.asarray(page) @ self.prefix),
            depends_on=frozenset({"processor", "vision", "base_embedding"}),
            contract="synthetic-grid-v1",
        )

    def resume(self, state: CutState) -> np.ndarray:
        return normalize_rows(np.asarray(state.state) @ self.suffix)


rng = np.random.default_rng(0)
shared_prefix = rng.normal(size=(12, 16))
v1 = SyntheticAdapter(shared_prefix, rng.normal(size=(16, 8)))
v2 = SyntheticAdapter(shared_prefix, rng.normal(size=(16, 8)))
pages = {f"page-{i}": rng.normal(size=(20, 12)) for i in range(50)}
queries = [rng.normal(size=(4, 8)) for _ in range(25)]

# 1. What did the release change? Tensor names come from the v2 checkpoint header.
scope = inspect_adapter_tensor_keys(
    [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.custom_text_proj.lora_A.weight",
    ]
)
active = VersionManifest(
    source="pages:v1", processor="proc:v1", vision="vit:v1",
    base_embedding="emb:v1", adapter="lora:v1", projection="head:v1",
    index_policy="flat",
)
target = VersionManifest(
    **{**active.to_dict(), "adapter": "lora:v2", "projection": "head:v2"}
)
assert active.changed_components(target) == scope.changed_components

# 2. Is the cut worth it? Leverage bounds the saving; the planner charges storage.
leverage = cut_leverage(
    preprocess_seconds=0.13, prefix_seconds=0.93, suffix_seconds=0.19
)
decision = choose_materializations(
    (
        MaterializationOption(
            name="post_vision",
            depends_on=frozenset({"processor", "vision", "base_embedding"}),
            storage_bytes=400_000_000,
            replay_seconds=0.19 * len(pages),
            quality_fraction=0.997,
        ),
    ),
    (scope.to_update_scenario("lora-v2"),),
    raw_rebuild_seconds=(0.13 + 0.93 + 0.19) * len(pages),
    storage_budget_bytes=500_000_000,
)
route = decision.routes[0].source

# 3. Build the target index from the stored cut and check it against raw v2.
stored = {k: v1.emit_cut(p, "post_vision") for k, p in pages.items()}
replayed = LateInteractionIndex((k, v2.resume(s)) for k, s in stored.items())
exact = LateInteractionIndex((k, v2.encode(p)) for k, p in pages.items())
stale = LateInteractionIndex((k, v1.encode(p)) for k, p in pages.items())


def ta_at_5(index: LateInteractionIndex) -> float:
    return float(
        np.mean(
            [
                target_agreement(
                    [r.item_id for r in exact.search(q, top_k=5)],
                    [r.item_id for r in index.search(q, top_k=5)],
                    k=5,
                )
                for q in queries
            ]
        )
    )


with tempfile.TemporaryDirectory(prefix="reprforge-") as tmp:
    deployment = Path(tmp)
    generation = deployment / "generations" / "lora-v2"
    save_index(generation / "index", replayed, target, source=route)
    seal_generation(
        deployment,
        "lora-v2",
        ("index/vectors.npz", "index/manifest.json"),
        version=target,
    )
    publish_generation(deployment, "lora-v2")
    active_path, manifest = resolve_active_generation(deployment)

    print(f"changed components : {sorted(scope.changed_components)}")
    print(f"post-vision cut legal: {scope.post_vision_cut_legal}")
    print(f"cut leverage        : {leverage:.2f}")
    print(f"planner route       : {route} (saving {decision.saving_fraction:.0%})")
    print(f"TA@5 exact cut      : {ta_at_5(replayed):.3f}")
    print(f"TA@5 stale index    : {ta_at_5(stale):.3f}")
    published = f"{active_path.name} -> adapter {manifest.version.adapter}"
    print(f"published generation: {published}")
