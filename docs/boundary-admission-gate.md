# Boundary-Weighted Representation Admission Gate

## Outcome

The first cutoff-aware mechanism passes its **offline** gate on IRPAPERS.
This is the first result in the project where a proposed policy beats a strong
cheap policy at the same train-only risk target.  It is not yet an A100 timing
result or a complete paper claim.

The full K=10 BM25+ColPali reference reaches 82.22% Recall@5.  Under a target
of 80% exact Top-5 agreement on historical queries:

| Admission policy | Held-out Recall@5 | Episode pages retained | Visual events | Exact Top-5 agreement |
|---|---:|---:|---:|---:|
| Frequency only | 81.67% | 474 / 680 (69.7%) | 1,594 | 82.22% |
| **Boundary weighted** | **83.33%** | **389 / 680 (57.2%)** | **1,480** | 79.44% |
| Full K=10 reference | 82.22% | all candidates | 1,800 | 100% |

Relative to the eligible pages in five independent held-out workload
episodes, boundary weighting removes 42.8% of visual page construction.  It
uses 85 fewer pages and 114 fewer query-page scoring events than frequency at
the same training target.  Recall@5 is two queries higher than the full K=10
reference and three higher than frequency; with one gold page per query, these
small differences must not be presented as statistically robust improvements.

The preregistered requirements were: lose no more than one Recall@5 query,
remove at least 20% of unique pages, use fewer pages than frequency admission,
and do no worse than frequency on Recall@5.  All four pass.

## Algorithm candidate

The mechanism is not “choose a better K.”  It treats visual representation as
a workload-level physical-design resource.

### Historical calibration

For every BM25 rank `j` in a Top-10 cohort, compute

```text
flip_risk[j] = P(candidate membership at Top-5 changes
                 after full visual evidence)
```

The labels compare the cheap and fully observed rankings.  They do not use
qrels.  The same logs provide a rank-conditioned prior for the normalized
visual score of an unbuilt candidate.

### Workload compilation

For every page `p` appearing in a new query episode, assign

```text
admission_value[p] = sum over occurrences (flip_risk[BM25_rank(p)])
```

Frequency admission gives every occurrence weight one.  Boundary admission
instead values appearances near positions where visual evidence historically
changes Top-5 membership.  It therefore avoids spending equal representation
capacity on an obvious head result, a plausible boundary challenger and a
hopeless tail result.

The compiler searches a small budget grid and chooses the lowest historical
budget that reaches the requested teacher-agreement target.  Pages with the
largest aggregate admission value receive visual representations.

### Query execution

Selected pages receive actual ColPali scores, normalized over the selected
pages for that query.  Missing pages use the train-only rank prior.  Those
scores are fused with candidate-relative BM25 scores and the resulting Top-5
is returned.

## Why this is more than a threshold

The policy connects three levels that fixed K conflates:

1. **decision risk:** where a candidate can change the requested cutoff;
2. **workload reuse:** how many queries can benefit from one built page; and
3. **physical action:** which page representations are actually admitted.

The earlier tail-only idea optimized the first level but failed the systems
test: skipping BM25 ranks 1--2 reduced scoring events by 20% but unique pages
by only 3.5%, because head pages are highly reused.  Boundary-weighted
admission explicitly multiplies decision value by occurrences, so it balances
uncertainty against amortization.

## Evaluation discipline

The 19 query-source papers are assigned intact to five deterministically
balanced folds.  The held-out fold is pooled into one ordinary workload
episode.  Source-paper identity is used only to prevent train/test leakage;
the compiler sees only BM25 candidate occurrences for held-out queries and
does not know which paper or page is relevant.

The chosen budget is also train-only.  Four folds choose 60% and one chooses
50% for boundary admission; frequency chooses 70% in every fold.  Qrels are
opened only after the plans and rankings are frozen.

## What is still missing

1. The 389/680 count sums five independent test episodes.  It is not the size
   of one global persistent cache, and shared pages across folds are charged
   independently by design.
2. Candidate-event and page-count reductions are not wall-clock speedups.  In
   the measured K=10 run, visual construction dominates 118 seconds while
   MaxSim scoring is below one second.  The next executor must therefore prove
   fewer physical encodes, not only fewer score operations.
3. Calibration currently requires historical full-visual score logs.  A real
   system must amortize sparse shadow scoring or periodic recalibration rather
   than prebuild the full index merely to learn how not to build it.
4. The 80% agreement target is empirical, not a conformal guarantee, and
   held-out exact agreement falls to 79.44%.
5. IRPAPERS has one gold page per query and no temporal trace.  Transfer to a
   graded ViDoRe workload and an online query order is required.

## Next gate

Implement the plan in the resident compiler and replay each held-out fold as a
cold workload episode on one A100.  The gate is:

- at least 20% fewer physical page encodes than fixed K=10;
- Recall@5 loss no larger than one query;
- at least 1.15x lower construction-plus-retrieval time;
- no regression versus frequency-only admission at the same train target;
- identical results across two deterministic query orders, except timing.

If page construction does not fall proportionally, or batching destroys the
expected latency gain, this mechanism remains an offline allocator rather than
the core of the system paper.

## Reproduction

```bash
PYTHONPATH=. python -m tools.analyze_boundary_admission \
  --score-surface /path/to/summary-runtime-score-surface.npz \
  --queries /path/to/queries.csv \
  --output /tmp/boundary-admission.json
```

The committed compact result is
`results/systems/boundary-admission-gate.json`.
