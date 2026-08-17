"""Minimal model-agnostic ReprForge flow using synthetic endpoint vectors."""

import numpy as np

from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler


rng = np.random.default_rng(0)
compact_canary = [rng.normal(size=(8, 16)) for _ in range(4)]
full_canary_in_compact_slots = [page + 0.05 for page in compact_canary]

compiler = ReprForgeCompiler(
    CompilerConfig(
        profile=BackboneProfile(
            name="my-multi-vector-vlm",
            total_layers=18,
            split_after_layer=12,
            full_visual_tokens=16,
            compact_visual_tokens=8,
        ),
        rank=4,
        fit_steps=25,
    )
)
compiler.fit(compact_canary, full_canary_in_compact_slots)

pages = [(f"page-{index}", rng.normal(size=(8, 16))) for index in range(20)]
index = compiler.build(pages)
query = rng.normal(size=(6, 16))
candidates = index.search(query, top_k=5)

# A real integration decodes these pages and runs the original Full encoder.
full_vectors = {item_id: rng.normal(size=(16, 16)) for item_id, _ in pages}
ranking = index.refine(query, candidates, full_vectors.__getitem__, top_k=3)

for result in ranking:
    print(f"{result.item_id}: {result.score:.3f}")
