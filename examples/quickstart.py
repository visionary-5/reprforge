"""Executable end-to-end reference flow with a synthetic model adapter."""

from __future__ import annotations

import numpy as np

from reprforge import (
    BackboneProfile,
    BoundaryState,
    CoalescedState,
    CompilerConfig,
    ReprForgeCompiler,
    normalize_rows,
    refine_candidates,
)


class SyntheticAdapter:
    """Stand-in for a ColPali-style prefix/suffix model integration."""

    def __init__(self, profile: BackboneProfile) -> None:
        self.profile = profile
        self.projection = np.random.default_rng(9).normal(size=(16, 8))

    def run_prefix(
        self, document: object, *, split_after_layer: int
    ) -> BoundaryState:
        if split_after_layer != self.profile.split_after_layer:
            raise ValueError("unexpected split")
        return BoundaryState(
            hidden_states=np.asarray(document),
            grid_shape=(4, 4),
            auxiliary_tokens=2,
        )

    def run_suffix(
        self, boundary: BoundaryState, compact: CoalescedState
    ) -> np.ndarray:
        del boundary
        return normalize_rows(compact.hidden_states @ self.projection)


rng = np.random.default_rng(0)
profile = BackboneProfile(
    name="synthetic-multi-vector-vlm",
    total_layers=18,
    split_after_layer=9,
    full_visual_tokens=16,
    compact_visual_tokens=8,
)
compiler = ReprForgeCompiler(CompilerConfig(profile=profile, grid_shape=(4, 4)))
adapter = SyntheticAdapter(profile)
documents = [
    (f"page-{index}", rng.normal(size=(18, 16))) for index in range(20)
]
index = compiler.build_documents(adapter, documents)

query = rng.normal(size=(6, 8))
candidates = index.search(query, top_k=5)
full_vectors = {
    item_id: normalize_rows(rng.normal(size=(16, 8)))
    for item_id, _ in documents
}
ranking = refine_candidates(
    index,
    query,
    candidates,
    full_vectors.__getitem__,
    top_k=3,
)

for result in ranking:
    print(f"{result.item_id}: {result.score:.3f}")
