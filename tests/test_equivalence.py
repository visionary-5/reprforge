import numpy as np
import pytest

from reprforge import (
    ComponentEquivalence,
    ComponentOutputFingerprint,
    VersionManifest,
    certify_component_equivalence,
    certify_component_fingerprints,
    fingerprint_component_outputs,
)


def manifest(*, processor: str, adapter: str = "adapter-v1") -> VersionManifest:
    return VersionManifest(
        source="source",
        processor=processor,
        vision="vision",
        base_embedding="embedding",
        adapter=adapter,
        projection="projection",
        index_policy="policy",
    )


def outputs(offset: int = 0):
    return [
        {
            "input_ids": np.array([1, 2], dtype=np.int64),
            "pixels": np.arange(12, dtype=np.float32).reshape(3, 2, 2) + offset,
        },
        {
            "input_ids": np.array([3], dtype=np.int64),
            "pixels": np.ones((3, 1, 1), dtype=np.float32),
        },
    ]


def test_exact_outputs_create_collection_scoped_certificate():
    certificate = certify_component_equivalence(
        component="processor",
        source_fingerprint="processor-file-v1",
        target_fingerprint="processor-file-v2",
        scope_fingerprint="corpus-313-docs",
        source_outputs=outputs(),
        target_outputs=outputs(),
    )

    assert certificate.compared_items == 2
    assert certificate.output_fields == ("input_ids", "pixels")
    assert len(certificate.output_sha256) == 64
    assert ComponentEquivalence.from_dict(certificate.to_dict()) == certificate


def test_persisted_output_fingerprints_can_be_certified_later():
    source = fingerprint_component_outputs(
        scope_fingerprint="corpus",
        outputs=outputs(),
    )
    target = fingerprint_component_outputs(
        scope_fingerprint="corpus",
        outputs=outputs(),
    )
    restored = ComponentOutputFingerprint.from_dict(source.to_dict())

    certificate = certify_component_fingerprints(
        component="processor",
        source_component_fingerprint="processor-v1",
        target_component_fingerprint="processor-v2",
        source_outputs=restored,
        target_outputs=target,
    )
    assert certificate.compared_items == 2
    assert certificate.output_sha256 == source.output_sha256


def test_output_difference_fails_closed():
    with pytest.raises(ValueError, match="outputs differ"):
        certify_component_equivalence(
            component="processor",
            source_fingerprint="processor-file-v1",
            target_fingerprint="processor-file-v2",
            scope_fingerprint="corpus",
            source_outputs=outputs(),
            target_outputs=outputs(offset=1),
        )


def test_certificate_removes_only_proven_invalidation():
    old = manifest(processor="processor-file-v1")
    target = manifest(processor="processor-file-v2", adapter="adapter-v2")
    certificate = certify_component_equivalence(
        component="processor",
        source_fingerprint=old.processor,
        target_fingerprint=target.processor,
        scope_fingerprint="corpus",
        source_outputs=outputs(),
        target_outputs=outputs(),
    )

    assert old.changed_components(target) == frozenset({"processor", "adapter"})
    assert old.invalidated_components(
        target,
        scope_fingerprint="corpus",
        equivalences=(certificate,),
    ) == frozenset({"adapter"})
    scenario = old.invalidation_scenario(
        target,
        "adapter-update",
        scope_fingerprint="corpus",
        equivalences=(certificate,),
    )
    assert scenario.changed_components == frozenset({"adapter"})


def test_stale_scope_or_fingerprint_is_rejected():
    old = manifest(processor="processor-file-v1")
    target = manifest(processor="processor-file-v2", adapter="adapter-v2")
    certificate = certify_component_equivalence(
        component="processor",
        source_fingerprint=old.processor,
        target_fingerprint=target.processor,
        scope_fingerprint="corpus-a",
        source_outputs=outputs(),
        target_outputs=outputs(),
    )

    with pytest.raises(ValueError, match="does not cover"):
        old.invalidated_components(
            target,
            scope_fingerprint="corpus-b",
            equivalences=(certificate,),
        )


def test_equivalent_only_change_requires_no_rebuild_scenario():
    old = manifest(processor="processor-file-v1")
    target = manifest(processor="processor-file-v2")
    certificate = certify_component_equivalence(
        component="processor",
        source_fingerprint=old.processor,
        target_fingerprint=target.processor,
        scope_fingerprint="corpus",
        source_outputs=outputs(),
        target_outputs=outputs(),
    )

    assert not old.invalidated_components(
        target,
        scope_fingerprint="corpus",
        equivalences=(certificate,),
    )
    with pytest.raises(ValueError, match="equivalent on the certified scope"):
        old.invalidation_scenario(
            target,
            "processor-metadata-only",
            scope_fingerprint="corpus",
            equivalences=(certificate,),
        )
