"""Minimal ReprForge flow using synthetic hidden states and endpoints."""

import numpy as np

from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler

rng = np.random.default_rng(0)

compiler = ReprForgeCompiler(
    CompilerConfig(
        profile=BackboneProfile(
            name="my-multi-vector-vlm",
            total_layers=18,
            split_after_layer=12,
            full_visual_tokens=16,
            compact_visual_tokens=8,
        ),
    )
)

# A real adapter obtains this tensor at the selected document-encoder boundary,
# then continues the original suffix using result.plan.compact_positions().
layer_hidden = rng.normal(size=(18, 16))  # 4x4 visual grid + two auxiliaries
result = compiler.compile_hidden_state(
    layer_hidden,
    grid_shape=(4, 4),
    auxiliary_tokens=2,
)
assert result.hidden_states.shape == (10, 16)

# Here random matrices stand in for compact suffix endpoints from 20 pages.
pages = [(f"page-{index}", rng.normal(size=(10, 16))) for index in range(20)]
index = compiler.build(pages)
query = rng.normal(size=(6, 16))
candidates = index.search(query, top_k=5)

# A real integration decodes these pages and runs the original Full encoder.
full_vectors = {item_id: rng.normal(size=(16, 16)) for item_id, _ in pages}
ranking = index.refine(query, candidates, full_vectors.__getitem__, top_k=3)

for result in ranking:
    print(f"{result.item_id}: {result.score:.3f}")
