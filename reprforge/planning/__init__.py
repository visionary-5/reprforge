"""Logical and physical plans for representation lifecycles."""

from .equivalence import (
    ComponentEquivalence,
    ComponentOutputFingerprint,
    certify_component_equivalence,
    certify_component_fingerprints,
    fingerprint_component_outputs,
)
from .materialization import (
    MaterializationDecision,
    MaterializationOption,
    UpdateRoute,
    UpdateScenario,
    choose_materializations,
    evaluate_materializations,
)
from .models import BackboneProfile, CompilePlan
from .planner import plan_topology_anchored
from .versions import VersionManifest

__all__ = [
    "BackboneProfile",
    "ComponentEquivalence",
    "ComponentOutputFingerprint",
    "CompilePlan",
    "MaterializationDecision",
    "MaterializationOption",
    "UpdateRoute",
    "UpdateScenario",
    "VersionManifest",
    "certify_component_equivalence",
    "certify_component_fingerprints",
    "choose_materializations",
    "evaluate_materializations",
    "fingerprint_component_outputs",
    "plan_topology_anchored",
]
