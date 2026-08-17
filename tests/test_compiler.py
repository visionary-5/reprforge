import numpy as np
import pytest

from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler


def test_compiler_builds_searchable_index_after_query_free_fit() -> None:
    rng = np.random.default_rng(3)
    compact_canary = [rng.normal(size=(4, 8)) for _ in range(4)]
    full_canary = [page + 0.05 for page in compact_canary]
    compiler = ReprForgeCompiler(
        CompilerConfig(
            profile=BackboneProfile(
                name="test-vlm",
                total_layers=18,
                split_after_layer=12,
                full_visual_tokens=8,
                compact_visual_tokens=4,
            ),
            rank=2,
            fit_steps=5,
        )
    )

    compiler.fit(compact_canary, full_canary)
    index = compiler.build(
        [("page-1", compact_canary[0]), ("page-2", compact_canary[1])]
    )

    assert len(index) == 2
    assert index.search(compact_canary[0], top_k=1)[0].item_id == "page-1"


def test_compiler_requires_fit_before_build() -> None:
    compiler = ReprForgeCompiler(
        CompilerConfig(
            profile=BackboneProfile("test", 4, 2, 8, 4),
            rank=2,
        )
    )
    with pytest.raises(RuntimeError, match="fit the compiler"):
        compiler.build([("page", np.ones((4, 4)))])
