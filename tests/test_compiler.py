import numpy as np
import pytest

from reprforge import BackboneProfile, CompilerConfig, ReprForgeCompiler


def test_compiler_builds_searchable_index_from_compact_endpoints() -> None:
    rng = np.random.default_rng(3)
    pages = [rng.normal(size=(4, 8)) for _ in range(2)]
    compiler = ReprForgeCompiler(
        CompilerConfig(
            profile=BackboneProfile(
                name="test-vlm",
                total_layers=18,
                split_after_layer=12,
                full_visual_tokens=8,
                compact_visual_tokens=4,
            )
        )
    )

    index = compiler.build([("page-1", pages[0]), ("page-2", pages[1])])

    assert len(index) == 2
    assert index.search(pages[0], top_k=1)[0].item_id == "page-1"


def test_compiler_checks_hidden_grid_against_profile() -> None:
    compiler = ReprForgeCompiler(
        CompilerConfig(profile=BackboneProfile("test", 4, 2, 16, 8))
    )
    result = compiler.compile_hidden_state(
        np.arange(18 * 4, dtype=float).reshape(18, 4) + 1,
        grid_shape=(4, 4),
        auxiliary_tokens=2,
    )
    assert result.hidden_states.shape == (10, 4)
    assert result.plan.persistent_fraction == pytest.approx(0.5)

    with pytest.raises(ValueError, match="full capacity"):
        compiler.compile_hidden_state(
            np.ones((6, 4)), grid_shape=(2, 2), auxiliary_tokens=2
        )
