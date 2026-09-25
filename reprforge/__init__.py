"""ReprForge: validated intermediate-state reuse for retriever upgrades.

Public surface, one module per concern:

* :mod:`reprforge.versions`     -- the version tuple and its dependency delta
* :mod:`reprforge.dependencies` -- classify adapter tensors by encoder stage
* :mod:`reprforge.equivalence`  -- collection-scoped output certificates
* :mod:`reprforge.planning`     -- cut selection, leverage and break-even
* :mod:`reprforge.adapter`      -- the emit-cut / resume contract for backbones
* :mod:`reprforge.index`        -- MaxSim, Target Agreement, checksummed storage
* :mod:`reprforge.generation`   -- immutable generations and atomic publication
"""

from .adapter import CutState, DocumentEncoderAdapter
from .dependencies import (
    AdapterDependencyScope,
    classify_tensor,
    inspect_adapter_tensor_keys,
)
from .equivalence import (
    ComponentEquivalence,
    ComponentOutputFingerprint,
    certify_component_equivalence,
    certify_component_fingerprints,
    fingerprint_component_outputs,
)
from .generation import (
    GenerationArtifact,
    GenerationManifest,
    publish_generation,
    resolve_active_generation,
    seal_generation,
    validate_generation,
)
from .index import (
    IndexManifest,
    LateInteractionIndex,
    SearchResult,
    load_index,
    maxsim_score,
    normalize_rows,
    save_index,
    target_agreement,
)
from .page_reuse import PageReuseDecision, assess_page_reuse
from .planning import (
    MaterializationDecision,
    MaterializationOption,
    UpdateRoute,
    UpdateScenario,
    break_even_upgrades,
    choose_materializations,
    cut_leverage,
    evaluate_materializations,
)
from .versions import COMPONENTS, VersionManifest

__all__ = [
    "COMPONENTS",
    "AdapterDependencyScope",
    "ComponentEquivalence",
    "ComponentOutputFingerprint",
    "CutState",
    "DocumentEncoderAdapter",
    "GenerationArtifact",
    "GenerationManifest",
    "IndexManifest",
    "LateInteractionIndex",
    "MaterializationDecision",
    "MaterializationOption",
    "PageReuseDecision",
    "SearchResult",
    "UpdateRoute",
    "UpdateScenario",
    "VersionManifest",
    "assess_page_reuse",
    "break_even_upgrades",
    "certify_component_equivalence",
    "certify_component_fingerprints",
    "choose_materializations",
    "classify_tensor",
    "cut_leverage",
    "evaluate_materializations",
    "fingerprint_component_outputs",
    "inspect_adapter_tensor_keys",
    "load_index",
    "maxsim_score",
    "normalize_rows",
    "publish_generation",
    "resolve_active_generation",
    "save_index",
    "seal_generation",
    "target_agreement",
    "validate_generation",
]

__version__ = "0.5.0"
