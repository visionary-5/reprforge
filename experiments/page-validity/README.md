# Page-scoped processor validity

This extends the existing collection-scoped component certificate. A processor
change need not affect every page. Compare complete processor outputs per page
and discharge only that page's processor mismatch. Other changed upstream
dependencies still force raw. No planner or approximate admission rule is added.

Implementation: [`../../reprforge/page_reuse.py`](../../reprforge/page_reuse.py).
Target integration: [`endpoint.py`](endpoint.py). The protocol was frozen before
CPU checks and GPU sample selection.

## Evidence

The unchanged independent-endpoint 200-page sample was processed at 12,845,056
and 602,112 pixels. These budgets match the recorded v0.1 → v0.2 budget delta;
other settings use the common pinned base processor. This isolates the budget
change; it does not certify the complete shipped processor contracts.

| Collection | Pages | Identical complete processor outputs |
|---|---:|---:|
| ArxivQA | 34 | 4 |
| DocVQA | 34 | 1 |
| InfoVQA | 33 | 1 |
| TabFQuAD | 33 | 9 |
| TAT-DQA | 33 | 0 |
| Shift | 33 | 0 |
| Total | 200 | 15 |

Collection-wide invalidation admits zero pages. Page evidence recovers 15/200
without changing inputs or target budget. The first eight matching and first
eight differing pages were checked on A100 / BF16 / batch 1 with the v0.2 target
adapter and previously persisted v0.1 states.

| Diagnostic route | Matching-input pages | Differing-input pages |
|---|---:|---:|
| Raw for all | Independent reference | Independent reference |
| Ignore processor change; replay all | 8/8 bitwise equal | 0/8 equal; all shapes differ |
| Page-scoped replay / raw fallback | 8/8 bitwise equal | 8/8 bitwise equal through separately executed raw |

All 16 routed outputs were finite. The rejected sample tests shape-changing
inputs, not every kind of same-shape processor change. Unit tests separately
cover pixel/token changes, signed zero, missing fields, wrong page identity,
vision changes and unchanged processor labels hiding changed inputs.

## Conditions and limitations

If a page's complete processed input is equal, remaining upstream computation
is unchanged under the numerical contract, and the stored state faithfully
represents that computation, a deterministic prefix produces the required
target state. The target suffix can then resume. This is a sufficient condition,
not a necessary condition or arbitrary model-equivalence proof. SHA-256 equality
assumes no digest collision. Integrations must declare complete dependencies
and fields and bind the input digest to the actual captured state.

The source states predate this mechanism. Processor fingerprints were rebuilt
retrospectively; original image/state hashes and frozen model artifacts were
checked. Deployment must record fingerprints during initial encoding. This
experiment does not validate that initial-capture workflow or end-to-end savings.

CPU totals: source preprocessing 49.236 s, source hashing 16.395 s; target
preprocessing 14.864 s, target hashing 2.276 s. These are descriptive single-pass
measurements. The recovered 7.5% page fraction is not a time-saving fraction;
small pages may contain disproportionately little visual computation. No ranking
experiment was run for this extension.

Reuse/change propagation has prior art, including
[Acar et al., A Consistent Semantics of Self-Adjusting Computation](https://arxiv.org/abs/1106.0478).
The claim here is a refinement of ReprForge's processor admission supported by
a mixed-validity boundary, not invention of incremental computation.

## Reproduce

First reproduce `independent-endpoint` to obtain source states, `manifest.json`,
`inputs.json`, and `source.json`. Use the pinned GPU environment.

```bash
python experiments/page-validity/probe.py \
  --prior /path/to/independent-endpoint-output --output /new/path/cpu
CUDA_VISIBLE_DEVICES=0 python experiments/page-validity/endpoint.py \
  --prior /path/to/independent-endpoint-output --probe /new/path/cpu \
  --output /new/path/gpu
```

Commands refuse existing output directories. Run GPU only if CPU finds both
matching and differing pages. Selection uses pre-GPU admission decisions and
is stratified; 16/16 is not a population accuracy estimate.

`cpu-summary.json` and `result.json` copy raw summaries exactly. The LFS bundle
`evidence/page-validity-evidence.tar.gz` preserves per-page CPU records, executed
source, protocol, GPU summary and log. Large tensor banks remain external and
are SHA-256 identified in `result.json`.
