"""The adapter contract: an exact cut must reproduce the raw route exactly."""

from __future__ import annotations

import numpy as np
import pytest

from reprforge import (
    CutState,
    DocumentEncoderAdapter,
    LateInteractionIndex,
    normalize_rows,
    target_agreement,
)


class SyntheticAdapter:
    """Two-stage encoder: a frozen 'visual prefix' and a version-specific suffix.

    ``prefix`` stands in for processor + vision tower + merger and is shared by
    both versions; ``suffix`` stands in for the decoder LoRA + terminal
    projection that a release changes.
    """

    cuts = ("post_vision",)

    def __init__(self, prefix: np.ndarray, suffix: np.ndarray) -> None:
        self.prefix = prefix
        self.suffix = suffix

    def _prefix(self, document: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(document) @ self.prefix)

    def encode(self, document: object) -> np.ndarray:
        return normalize_rows(self._prefix(np.asarray(document)) @ self.suffix)

    def emit_cut(self, document: object, cut: str) -> CutState:
        if cut != "post_vision":
            raise KeyError(cut)
        return CutState(
            cut=cut,
            state=self._prefix(np.asarray(document)),
            depends_on=frozenset({"processor", "vision", "base_embedding"}),
            contract="synthetic-grid-v1",
        )

    def resume(self, state: CutState) -> np.ndarray:
        state.validate()
        if state.contract != "synthetic-grid-v1":
            raise ValueError("resume contract mismatch")
        return normalize_rows(np.asarray(state.state) @ self.suffix)


def make_versions(seed: int = 0) -> tuple[SyntheticAdapter, SyntheticAdapter]:
    rng = np.random.default_rng(seed)
    prefix = rng.normal(size=(12, 16))
    old = SyntheticAdapter(prefix, rng.normal(size=(16, 8)))
    new = SyntheticAdapter(prefix, rng.normal(size=(16, 8)))
    return old, new


def test_synthetic_adapter_satisfies_the_protocol() -> None:
    old, _ = make_versions()
    assert isinstance(old, DocumentEncoderAdapter)


def test_exact_cut_replay_equals_raw_target_encoding() -> None:
    old, new = make_versions()
    rng = np.random.default_rng(1)
    pages = [rng.normal(size=(20, 12)) for _ in range(5)]

    for page in pages:
        state = old.emit_cut(page, "post_vision")  # materialised under v1
        replayed = new.resume(state)  # target suffix from the stored cut
        np.testing.assert_array_equal(replayed, new.encode(page))


def test_compressed_cut_is_admitted_by_agreement_not_by_construction() -> None:
    old, new = make_versions()
    rng = np.random.default_rng(2)
    pages = {f"p{i}": rng.normal(size=(20, 12)) for i in range(40)}
    queries = [rng.normal(size=(4, 8)) for _ in range(20)]

    target = LateInteractionIndex((k, new.encode(v)) for k, v in pages.items())
    stale = LateInteractionIndex((k, old.encode(v)) for k, v in pages.items())

    def lossy(state: CutState, scale: float) -> CutState:
        quantised = np.round(np.asarray(state.state) / scale) * scale
        return CutState(state.cut, quantised, state.depends_on, state.contract)

    fine = LateInteractionIndex(
        (k, new.resume(lossy(old.emit_cut(v, "post_vision"), 1e-3)))
        for k, v in pages.items()
    )
    coarse = LateInteractionIndex(
        (k, new.resume(lossy(old.emit_cut(v, "post_vision"), 1.0)))
        for k, v in pages.items()
    )

    def ta(index: LateInteractionIndex) -> float:
        return float(
            np.mean(
                [
                    target_agreement(
                        [r.item_id for r in target.search(q, top_k=5)],
                        [r.item_id for r in index.search(q, top_k=5)],
                        k=5,
                    )
                    for q in queries
                ]
            )
        )

    assert ta(target) == 1.0
    assert ta(fine) >= ta(coarse)
    assert ta(fine) > ta(stale)


def test_cut_state_validation() -> None:
    with pytest.raises(ValueError, match="cut name"):
        CutState("", None, frozenset({"vision"}), "c").validate()
    with pytest.raises(ValueError, match="dependencies"):
        CutState("post_vision", None, frozenset(), "c").validate()
    with pytest.raises(ValueError, match="contract"):
        CutState("post_vision", None, frozenset({"vision"}), "").validate()
