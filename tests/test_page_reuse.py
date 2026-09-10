from dataclasses import replace

import numpy as np
import pytest

from reprforge.equivalence import fingerprint_component_outputs
from reprforge.page_reuse import assess_page_reuse
from reprforge.versions import VersionManifest

SOURCE = VersionManifest("pages", "p0", "vision", "embedding", "a0", "head", "i")
TARGET = replace(SOURCE, processor="p1", adapter="a1")
FIELDS = frozenset({"pixels", "ids", "grid"})


def evidence(*, pixels=0.0, ids=1, page="page0", omit=None):
    values = {
        "pixels": np.array([pixels]),
        "ids": np.array([ids]),
        "grid": np.array([1, 2, 2]),
    }
    if omit:
        del values[omit]
    return fingerprint_component_outputs(scope_fingerprint=page, outputs=[values])


def decide(a, b, target=TARGET):
    return assess_page_reuse(
        SOURCE,
        target,
        page_fingerprint="page0",
        dependencies=frozenset({"source", "processor", "vision"}),
        required_processor_fields=FIELDS,
        source_outputs=a,
        target_outputs=b,
    )


def test_mixed_pages_recover_only_unchanged_inputs():
    assert decide(evidence(), evidence()).replay
    changed = decide(evidence(), evidence(pixels=1.0))
    assert not changed.replay
    assert changed.blockers == {"processor"}


def test_matching_grid_does_not_certify_pixels_or_text():
    assert not decide(evidence(), evidence(ids=2)).replay
    assert not decide(evidence(), evidence(pixels=-0.0)).replay


def test_processor_equivalence_cannot_discharge_vision_change():
    decision = decide(evidence(), evidence(), replace(TARGET, vision="new"))
    assert not decision.replay
    assert decision.blockers == {"vision"}


def test_same_version_label_cannot_hide_observed_input_change():
    assert not decide(evidence(), evidence(pixels=2), SOURCE).replay


def test_missing_fields_and_wrong_page_fail_closed():
    with pytest.raises(ValueError, match="complete"):
        decide(evidence(omit="ids"), evidence(omit="ids"))
    with pytest.raises(ValueError, match="exactly this one page"):
        decide(evidence(), evidence(page="another"))


def test_cannot_use_collection_certificate_for_a_page():
    multi = replace(evidence(), compared_items=200)
    with pytest.raises(ValueError, match="exactly this one page"):
        decide(multi, multi)
