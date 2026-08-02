# Representation View Control-Plane Result

## Outcome

The first database-style candidate representation control plane is implemented
and passes its CPU scale gate. The result establishes that catalog size and
planning overhead do not block the research direction. It does not establish
retrieval quality or GPU speedup because the scale workload uses synthetic
latent utility.

## Implemented system

`reprforge/representation_views.py` provides:

- metadata-only hypothetical representation views;
- budgeted optimistic probe scheduling;
- probe verification and rejection;
- budgeted materialization planning;
- legal lifecycle transitions and interrupted-work recovery;
- atomic catalog snapshots and deterministic reload;
- executor result validation before publication.

The BM25 cohort compiler and official ViDoRe adapter now accept a published
admission plan. Only admitted pages are visually encoded and scored; logical
BM25 candidate membership remains unchanged. Metrics separately report all
candidate events, admitted events, physical encodes and score pairs.

## Scale measurements

The deterministic seed is `20260802`. Each workload exposes three candidate
routes (`pool-9`, `pool-25`, `full-visual`) for pages observed in a skewed query
stream.

| Profile | Candidate events | Candidate pages | Candidate views | Probe views | Materialized views | Generation | Planning | Peak Python memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IRPAPERS-like | 1,800 | 607 | 1,821 | 438 | 205 | 0.040 s | 0.006 s | 5.2 MB |
| ViDoRe-v3-like | 62,000 | 10,198 | 30,594 | 7,036 | 2,992 | 1.169 s | 0.100 s | 80.0 MB |

The IRPAPERS-like catalog snapshot is 1.08 MB. The ViDoRe-v3-like snapshot is
18.21 MB. Repeating the first profile produces the same catalog SHA-256
`55103f7f43d260f083bfe5ab89cb4ecf9ced12e8573929a1e9e8063c52c3d198`
and identical state counts.

All 93 tests pass, with seven optional-dependency tests skipped.

## Design review

The positive result is architectural: candidate views can remain metadata-only
until verification, and the existing retrieval executor can honor a published
plan rather than eagerly building every candidate representation. Planning is
small compared with the previously measured tens to hundreds of seconds of
visual encoding.

The current planner is not a paper contribution. Its utility values are
supplied by the caller, and the scale benchmark uses synthetic latent values.
Three research debts dominate:

1. **What-if estimation.** Historical full-visual logs are currently needed to
   estimate boundary risk. Building those logs once merely to avoid future
   builds may erase the saving. Sparse shadow probes or transferable priors are
   required.
2. **Score calibration.** Candidate-relative fusion needs at least two observed
   visual candidates. A single observed page has no stable local z-score. The
   next mechanism needs a calibrated cross-query score model or a probe unit
   that guarantees a comparison pair.
3. **Probe reuse.** A page-level probe is useful only if the resulting artifact
   can be reused by later queries or by final materialization. Otherwise probe
   cost becomes pure overhead and a transient cascade may dominate.

The next experiment must therefore test prediction and artifact reuse on a
real public score surface, not increase catalog scale further.

## Reproduction

```bash
python -m reprforge.benchmark_representation_views \
  --items 3230 --queries 180 --candidate-k 10 --skew 1.1 \
  --output /tmp/reprforge-view-irpapers.json \
  --catalog /tmp/reprforge-view-irpapers-catalog.json

python -m reprforge.benchmark_representation_views \
  --items 26000 --queries 3100 --candidate-k 20 --skew 1.1 \
  --output /tmp/reprforge-view-vidore-v3.json \
  --catalog /tmp/reprforge-view-vidore-v3-catalog.json
```

The compact committed summary is
`results/systems/representation-view-control-plane.json`.
