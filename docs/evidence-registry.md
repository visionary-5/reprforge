# ReprForge evidence registry

Status: authoritative claim index, 2026-08-07.

This file separates reusable evidence from historical exploration. It does not
delete negative results; it prevents an old experiment from silently becoming
the current paper claim.

## Status vocabulary

- **active evidence**: directly informs the current problem or next protocol;
- **valid negative**: reproducible result that closes a particular method;
- **infrastructure**: correct implementation or measurement substrate without
  a novelty claim;
- **historical**: superseded research direction, retained for auditability;
- **awaiting physical reproduction**: score replay or accounting result that
  cannot support a final systems claim.

## Current problem evidence

| Artifact | Status | What it supports | What it does not support |
|---|---|---|---|
| `docs/dvi-page-verifier-gpu-pilot-v0-result-2026-08-07.md` | active evidence | Text and visual candidate generators have complementary failures; a common raw-page verifier can benefit from hybrid candidates. | Faithful DVI/AgenticOCR reproduction, partial-index cost, or end-to-end QA superiority. |
| `results/compiler-feasibility/residual-materialization-oracles/residual-materialization-oracle-v1/` | active evidence | Future-aware visual value is concentrated on a small page set in Pharma and Industrial. | A deployable selector; the oracle reads future labels/scores. |
| `docs/representation-value-sparsity-atlas-v0-result-2026-08-07.md` | active evidence | Visual contribution is signed: some pages help some queries and harm others; static complexity is weak. | That a particular risk feature predicts future benefit. |
| `docs/representation-granularity-audit-v0-result-2026-08-07.md` | valid negative | Mechanical fixed/XY-cut sub-page regions increase representation work without consistent benefit. | That all layout-aware or semantic region organization is unhelpful. |
| `docs/defer-materialize-phase-diagram-v0-result-2026-08-07.md` | awaiting physical reproduction | A preliminary cost-accounting boundary and several missing variables. | Direction-level GO/NO-GO, real DVI latency, or a realizable middle region. |
| `docs/omni-col-press-full-domain-cascade-result-2026-08-05.md` | active evidence | Compact locator plus Full reranking repairs mainly candidate-internal ranking errors across four domains. | That query Top-k routing is the paper contribution. |

## Closed or limited method branches

| Artifact family | Status | Recorded conclusion |
|---|---|---|
| Typed-Capacity V1 (`docs/results.md`) | historical | Passed storage gate but missed the preregistered quality gain and stable document-macro dominance. |
| Per-page intervention/utility estimators | valid negative | Rank interaction is strongly non-additive; independent page utility is not a reliable main abstraction. |
| Risk-limited query depth and RBRC | valid negative for main contribution | Quality control is useful, but savings over robust fixed Top-k were small/inconsistent and primarily query-side. |
| Elastic two-state cache V4 | valid negative | Apparent wins do not survive capacity-aware GDSF; binary cache residency is infrastructure. |
| Compression-risk certificate/physical dual-view bank | valid negative | Development certificates produced false-safe validation decisions; physical bank stayed near 90% of Full bytes. |
| Fixed-rate pooling/H-Pool/AGC | infrastructure and baseline | Useful compact locators and representation operators, not ReprForge novelty. |

## Reusable systems infrastructure

| Component | Status | Reuse in current protocol |
|---|---|---|
| OmniColPress pinned wrappers and patches | infrastructure | Full/compact physical build, scoring, and timing on A100/RTX 5090. |
| DVI-like Qwen2.5-VL page verifier | infrastructure | Common transient raw-page analysis stage; timing scope must be expanded. |
| BM25 and visual ranking adapters | infrastructure | Candidate coverage, fusion, and fair equal-budget comparisons. |
| Candidate representation catalog and atomic generations | infrastructure | Later online promotion execution after the static physical gate. |
| NumPy/PyTorch MaxSim and token-work scheduler | infrastructure | Correctness and serving-cost decomposition, not current novelty. |

## Evidence required next

The following cells are intentionally empty and are the reason no final method
claim is made yet:

1. physical partial-index curves at 1/2/5/10/20/40% page budgets;
2. strong text/structure/dense locator comparison, not BM25 alone;
3. realized selectors evaluated document/query-disjoint and on sealed domains;
4. defer versus persistent cumulative cost under locality, breadth, and drift;
5. complete raw-page I/O, preprocessing, H2D, VLM, search, and end-to-end p95/p99;
6. an external benchmark family and downstream answer/citation validation.

When a result is added, this registry must be updated in the same commit. Raw
result files remain immutable; interpretation can be corrected by a new note
and status change rather than rewriting history.
