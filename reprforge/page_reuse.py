"""Refine processor invalidation using complete, single-page output evidence.

This extends existing scoped equivalence to mixed collections. It does not
infer dependencies, certify a different vision model, or admit lossy states.
Integrations must bind the source fingerprint to the captured state, include
all processor outputs, and put numerical execution in the upstream contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .equivalence import (
    ComponentEquivalence,
    ComponentOutputFingerprint,
    certify_component_fingerprints,
)
from .versions import COMPONENTS, VersionManifest


@dataclass(frozen=True)
class PageReuseDecision:
    replay: bool
    blockers: frozenset[str]
    processor_equivalence: ComponentEquivalence | None = None


def assess_page_reuse(
    source: VersionManifest,
    target: VersionManifest,
    *,
    page_fingerprint: str,
    dependencies: frozenset[str],
    required_processor_fields: frozenset[str],
    source_outputs: ComponentOutputFingerprint,
    target_outputs: ComponentOutputFingerprint,
) -> PageReuseDecision:
    """Return replay or raw for one page; never extrapolate a sample certificate.

    Malformed evidence raises. Well-formed unequal outputs retain processor
    invalidation. Other changed dependencies always remain blockers. Even equal
    processor version labels cannot override observed unequal page inputs.
    """
    if not dependencies or not dependencies <= set(COMPONENTS):
        raise ValueError("cut dependencies must be known and non-empty")
    if "processor" not in dependencies:
        raise ValueError("this refinement requires a processor-dependent cut")
    if not page_fingerprint or not required_processor_fields:
        raise ValueError("page identity and complete output fields are required")
    for output in (source_outputs, target_outputs):
        output.validate()
        if output.scope_fingerprint != page_fingerprint or output.compared_items != 1:
            raise ValueError("evidence must cover exactly this one page")
        if frozenset(output.output_fields) != required_processor_fields:
            raise ValueError("evidence does not cover the complete processor contract")
    changed = source.changed_components(target)
    certificate = None
    equal = source_outputs.output_sha256 == target_outputs.output_sha256
    if equal and "processor" in changed:
        certificate = certify_component_fingerprints(
            component="processor",
            source_component_fingerprint=source.processor,
            target_component_fingerprint=target.processor,
            source_outputs=source_outputs,
            target_outputs=target_outputs,
        )
    remaining = source.invalidated_components(
        target,
        scope_fingerprint=page_fingerprint,
        equivalences=(certificate,) if certificate else (),
    )
    if not equal:
        remaining = remaining | {"processor"}
    blockers = frozenset(remaining & dependencies)
    return PageReuseDecision(not blockers, blockers, certificate)
