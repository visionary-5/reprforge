import numpy as np
import pytest

from reprforge import (
    BackboneProfile,
    BoundaryState,
    CoalescedState,
    CompilerConfig,
    ReprForgeCompiler,
)


class FakeAdapter:
    def __init__(self, profile: BackboneProfile) -> None:
        self.profile = profile

    def run_prefix(
        self, document: object, *, split_after_layer: int
    ) -> BoundaryState:
        assert split_after_layer == self.profile.split_after_layer
        return BoundaryState(document, (4, 4), auxiliary_tokens=2)

    def run_suffix(
        self, boundary: BoundaryState, compact: CoalescedState
    ) -> np.ndarray:
        del boundary
        return compact.hidden_states[:, :4]


def profile() -> BackboneProfile:
    return BackboneProfile("test-vlm", 18, 6, 16, 8)


def test_compiler_builds_collection_through_adapter_contract() -> None:
    rng = np.random.default_rng(3)
    compiler = ReprForgeCompiler(
        CompilerConfig(profile=profile(), grid_shape=(4, 4))
    )
    adapter = FakeAdapter(profile())
    documents = [
        ("page-1", rng.normal(size=(18, 8))),
        ("page-2", rng.normal(size=(18, 8))),
    ]

    index = compiler.build_documents(adapter, documents)

    assert len(index) == 2
    assert index.vector_count == 20
    assert index.search(index.records()[0][1], top_k=1)[0].item_id == "page-1"


def test_compiler_rejects_adapter_from_another_physical_plan() -> None:
    compiler = ReprForgeCompiler(
        CompilerConfig(profile=profile(), grid_shape=(4, 4))
    )
    other = BackboneProfile("other", 18, 6, 16, 8)

    with pytest.raises(ValueError, match="adapter profile"):
        compiler.compile_document(FakeAdapter(other), np.ones((18, 8)))
