# Progressive Evidence Acquisition: First Mechanism Probe

## Decision

Changing a fixed candidate depth `K` is not an algorithmic contribution.  This
probe replaces that control knob with a different system action:

> acquire expensive visual evidence in small batches, observe how it changes
> the ranking, and stop when the requested ranking decision is sufficiently
> stable.

The IRPAPERS result is a **positive mechanism signal for Top-1**, not yet a
complete algorithm.  In a leave-one-source-paper-out evaluation, the current
stability-plus-margin policy preserves the fixed Top-10 teacher's aggregate
Recall@1 while reducing visual candidate events by 44.9%.  It does not preserve
Recall@5.  The next design must therefore choose and certify evidence relative
to a requested cutoff, rather than add more hand-written thresholds.

## What the mechanism does

For each query, BM25 supplies an ordered Top-10 candidate set.  Visual evidence
is then acquired at depths 2, 4, 6, 8 and 10.  At each non-terminal stage, the
runtime observes two signals that were unavailable before visual work began:

1. whether the fused Top-1 identity stayed unchanged across two stages; and
2. the fused score margin between the current first and second candidates.

The runtime stops only if both conditions pass.  Each held-out source paper
uses thresholds selected on the other 18 papers.  Selection minimizes visual
page-events subject to zero disagreement with the fully observed Top-10
teacher on the training papers.  Qrels are not used to fit the thresholds;
they are used only for final Recall evaluation.

This distinction matters.  A static router predicts whether an expensive
reranker is useful from the query or BM25 output.  Progressive evidence
acquisition makes a sequence of smaller decisions after observing the actual
effect of the expensive modality.

## Controlled result

The score surface is the same A100 ColPali-v1.1 surface used by the independent
IRPAPERS transfer.  It contains 180 queries, 3,230 pages and 19 source papers.
The reference decision is candidate-relative BM25+ColPali fusion over the BM25
Top-10.

| Policy | Mean visual pages/query | Visual page-events | Unique pages | Top-1 disagreement vs K=10 | Recall@1 | Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.00 | 0 | 0 | 18.89% | 47.22% | 78.33% |
| Fixed K=4 | 4.00 | 720 | 228 | 10.56% | 47.22% | 78.33% |
| Fixed K=6 | 6.00 | 1,080 | 311 | 5.56% | 46.67% | 81.11% |
| Fixed K=8 | 8.00 | 1,440 | 406 | 2.78% | 48.33% | 81.67% |
| Fixed K=10 teacher | 10.00 | 1,800 | 511 | 0% | 48.89% | 82.22% |
| Naive winner stability | 4.46 | 802 | 251 | 6.67% | 47.22% | 76.67% |
| Paper-disjoint BM25-margin router | 7.72 | 1,390 | 418 | 0.56% | 48.89% | 81.67% |
| **Progressive stability + margin** | **5.51** | **992** | **309** | **1.11%** | **48.89%** | **78.33%** |
| Teacher-visible stage oracle | 2.51 | 452 | 178 | 0% | 48.89% | 79.44% |

The progressive policy chooses depth 4 for 101 queries, 6 for 35, 8 for 31 and
10 for only 13.  Relative to fixed K=10, this removes 808 candidate events and
202 unique visual pages.  It also requires 398 fewer events than the
paper-disjoint BM25-margin router.  This supports the mechanism claim that
**observed visual intervention contains decision information that the cheap
retriever alone does not expose**.

These counts are not yet a wall-clock speedup.  The resident implementation
deduplicates pages and batches encoder calls, so a real progressive executor
must be measured before translating event reduction into latency.

## The important failure

The same policy lowers Recall@5 from 82.22% to 78.33%, and its exact Top-5-set
agreement with the teacher is only 49.44%.  Preserving the winner is much
easier than establishing that no unobserved candidate can enter the requested
Top-5 set.  A stricter exploratory Top-5 stability rule preserved Recall@5 but
needed an average of about 9.56 of 10 pages, leaving only a 4.4% event saving.

This is the current research bottleneck, not threshold tuning:

> In what order should the system acquire unbuilt representations, and what
> low-cost bound can rule out an unobserved candidate crossing the requested
> rank boundary?

## Prior-art boundary

Several nearby formulations are already occupied:

- [Adaptive Re-Ranking (ICTIR 2026)](https://arxiv.org/abs/2606.25249)
  labels each query by qrel-derived quality and measured latency, then uses a
  BERT classifier to route among BM25, a light reranker and a heavy reranker.
  It chooses a complete path before reranking; it does not acquire individual
  document representations sequentially or use observed intervention to stop.
- [Certified Error Control of Candidate Set Pruning (EMNLP
  2022)](https://aclanthology.org/2022.emnlp-main.23/) calibrates a first-stage
  score threshold that reduces the candidate set while controlling a ranking
  loss.  It is the strongest conceptual baseline for future risk control, but
  it prunes before the expensive ranker and reports that tight guarantees
  usually require 1,000--10,000 calibration examples.
- [Light-ColPali/ColQwen2](https://arxiv.org/abs/2506.04997) compresses the
  patch embeddings stored for every visual page.  It shows that offline patch
  importance is query-dependent and that token merging is safer than pruning.
  ReprForge instead asks which page representation should be acquired at all;
  the two mechanisms are complementary.
- [Quake](https://arxiv.org/abs/2506.03437) adapts ANN partitions and search
  parameters under dynamic, skewed vector workloads.  It assumes vectors
  already exist and optimizes access to them; the expensive representation
  acquisition problem occurs before that index layer.
- [Tail-Aware Adaptive-k](https://arxiv.org/abs/2606.11907) selects how many
  retrieved contexts to send downstream from the shape of a completed score
  list.  It does not decide how many expensive representations must first be
  built to obtain that list.

The broader active-ranking literature supplies a useful design principle:
sample the candidates whose confidence intervals overlap the decision boundary
and stop when the Top-k set is separated.  Applying that principle to visual
document representation construction is the next algorithmic step; merely
renaming the current margin thresholds would not be a contribution.

## Next design: cutoff-aware active acquisition

The next prototype should replace prefix acquisition with three explicit
components:

1. **Boundary model.**  For every unobserved candidate, estimate an interval
   for its possible fused score using BM25 state, candidate rank, OCR/layout
   features and paper-disjoint calibration data.
2. **Acquisition function.**  Encode the candidate with the largest estimated
   probability of crossing the current requested cutoff, divided by its
   construction cost.  This tests whether acquisition order, not only stopping,
   creates material savings.
3. **Stopping certificate.**  Stop when no unobserved candidate's upper bound
   overlaps the current Top-r lower bound.  Evaluate r in {1, 5, 20}; report
   empirical risk separately from any formal guarantee.

The immediate falsification test is demanding but bounded: on held-out source
papers, the active ordering must preserve K=10 Recall@5 within one query
(0.56 point) while reducing visual page-events by at least 20%.  It must beat
fixed K, naive stability, the BM25-margin router and a certified-pruning-style
score threshold.  If it cannot, IRPAPERS does not support a useful listwise
acquisition algorithm with the available signals.

That offline gate has now been run. Boundary-weighted admission passes on five
source-paper-disjoint folds, retaining 57.2% of eligible episode pages and
reaching 83.33% Recall@5. The complete result and its remaining systems debts
are recorded in `boundary-admission-gate.md`.

## Reproduction

```bash
PYTHONPATH=. python -m tools.analyze_progressive_evidence \
  --score-surface /path/to/summary-runtime-score-surface.npz \
  --queries /path/to/queries.csv \
  --output /tmp/progressive-evidence.json
```

The full score surface remains outside Git because it can be regenerated from
the public benchmark and model.  Its SHA-256 is
`8a1553afbe257fc8590ac60a2b363a9672158b97957505925d9a36fa1d21718b`.
The committed compact result is
`results/systems/progressive-evidence-probe.json`.
