# Strong scheduler baselines and sensitivity replay

Date: 2026-08-04. Branch: `exp/scheduler-baselines`.

## Decision: GO

The cohort-frontier mechanism passes the strong-baseline gate on the recovered
current BM25/ColPali frozen surface:

- at the registered K=20, request-batch=8 setting, frontier has the lowest
  mean completion page-work among every feasible simple policy on both HR and
  Finance;
- neither static popularity, CaGR-style overlap grouping, shortest-missing,
  nor reuse-only reproduces the result;
- frontier beats every popularity/overlap-only explanation on mean work in
  all 12 K/batch settings in both domains;
- it is not Pareto-dominated by a non-oracle policy in 12/12 HR and 11/12
  Finance settings; the lone exception is Finance K=10, batch=1;
- an optimistic full-stream offline work refinement is only 1.4% better on HR
  and 1.2% better on Finance at the primary setting, leaving limited rather
  than structural work-optimization headroom.

This supports a narrow claim: resident-frontier information adds value beyond
static query grouping, popularity, or shortest remaining work.  It does not
prove online arrival robustness or algorithmic optimality.

## What was compared

All schedulers receive only the cheap locator's query--page candidate graph.
They cannot inspect relevance labels, visual scores, or answer outcomes.
Relevance labels are loaded only after scheduling to draw the nDCG--work
curve.

| Policy | Question isolated |
|---|---|
| FIFO | Does the official query order suffice? |
| Random, 10 seeds | Is the result an ordering accident? |
| Static popularity | Does full-stream page frequency alone explain it? |
| Overlap-only | Does CaGR-style static query grouping explain it? |
| Shortest-missing | Is shortest remaining cohort work sufficient? |
| Reuse-only | Is dynamic future popularity sufficient? |
| Frontier | Does completion distance plus resident/staged reuse help? |
| Offline work greedy | What does a full-stream, multi-start, qrel-free work optimizer achieve? |

The offline work greedy is not described as an exact oracle.  It takes the
best deterministic qrel-free starting schedule and repeatedly repartitions
adjacent batch pairs to reduce the exact sum of query completion page-work.
This provides an optimistic work-only lower envelope while remaining
computable at K up to 50 and batch size up to 16.

## Primary current result: K=20, request batch=8

### HR: BM25 locator and ColPali visual refinement

Candidate union: 895 pages.  Initial/final mean nDCG@10: 0.48600/0.53729.

| Schedule | Mean pages | P50 | P95 | nDCG--work AUC |
|---|---:|---:|---:|---:|
| FIFO | 718.60 | 768 | 892 | 0.502763 |
| Static popularity | 593.52 | 645 | 865 | **0.511895** |
| Overlap-only | 596.53 | 661 | 872 | 0.510576 |
| Shortest-missing | 583.35 | **627** | **855** | 0.510301 |
| Reuse-only | 645.82 | 726 | 886 | 0.509307 |
| **Frontier** | **579.83** | **627** | **855** | 0.510123 |
| Offline work greedy | **571.70** | **624** | **854** | 0.510020 |

Frontier cuts mean completion work by 19.3% versus FIFO, 2.3% versus static
popularity, 2.8% versus overlap-only, and 0.6% versus shortest-missing.  Static
popularity has higher post-hoc quality--work AUC, while frontier has lower
work; neither dominates the other.  The work result is within 1.4% of the
offline greedy lower envelope.

### Finance: BM25 locator and ColPali visual refinement

Candidate union: 1,855 pages.  Initial/final mean nDCG@10: 0.52852/0.56280.

| Schedule | Mean pages | P50 | P95 | nDCG--work AUC |
|---|---:|---:|---:|---:|
| FIFO | 1,208.73 | 1,341 | 1,811 | 0.548059 |
| Static popularity | 893.78 | 891 | 1,684 | 0.553009 |
| Overlap-only | 893.71 | 867 | 1,698 | 0.552353 |
| Shortest-missing | 901.16 | 837 | 1,698 | 0.552698 |
| Reuse-only | 1,004.13 | 1,033 | 1,750 | 0.551054 |
| **Frontier** | **861.37** | 824 | 1,698 | **0.553551** |
| Offline work greedy | **850.91** | **821** | **1,682** | 0.552950 |

Frontier cuts mean completion work by 28.7% versus FIFO, 3.6% versus static
popularity and overlap-only, and 4.4% versus shortest-missing.  It also has the
highest quality--work AUC, including against the work-only offline greedy.
Static popularity has a 14-page better P95, so the result is strong on mean,
median, and AUC but not every tail statistic.

## K and request-batch sensitivity

The registered grid is K in {10, 20, 50} crossed with request batch in
{1, 4, 8, 16}.

| Current surface | Frontier beats all popularity/overlap policies on mean work | Not Pareto-dominated by a non-oracle policy | Not dominated including offline greedy |
|---|---:|---:|---:|
| HR | 12/12 | 12/12 | 7/12 |
| Finance | 12/12 | 11/12 | 8/12 |

The single non-oracle domination occurs for Finance at K=10, batch=1, where
shortest-missing is slightly better on both work and quality AUC.  At the
paper setting K=20/batch=8 and at every K=20 setting, frontier survives in
both domains.  The offline greedy commonly improves work by construction but
does not consistently improve quality AUC, because it is label-free and
optimizes work only.

The sensitivity evidence supports this wording:

> Scheduling on the resident representation frontier reduces the construction
> work required to complete query cohorts beyond static overlap, page
> popularity, and shortest-missing baselines across two domains and a broad
> K/batch grid, while remaining within 1.5% of an optimistic offline work
> refinement at the primary setting.

Avoid saying that frontier universally dominates every metric or that the
offline combinatorial problem is solved.

## Provenance

The recovered current traces exactly match the hashes already recorded in the
candidate-fusion and progressive-quality results.

| Surface | HR text / visual SHA256 prefix | Finance text / visual SHA256 prefix | K=20 union |
|---|---|---|---:|
| Current BM25/ColPali | `f057...` / `8a6e...` | `55bf...` / `a6bb...` | 895 / 1,855 |
| Legacy heterogeneity stress trace | `6afe...` / `a516...` | `8fcf...` / `8756...` | 932 / 1,619 |

The machine-readable JSON records full SHA256 digests, source paths, schedule
order digests, and compacted trajectory digests.  Every replay asserts that
all schedules end with the identical candidate union.

The legacy surface remains in the branch as a representation-transfer stress
test, not as a replacement for current paper numbers.  Its primary K=20,
batch=8 result is directionally consistent: frontier beats the best simple
popularity/overlap work baseline by 4.0% on HR and 2.8% on Finance and is
within 1.7%/2.3% of the offline work greedy.

## Reproduction

No SSH, GPU, network access, or downloads are used.

```bash
PYTHONPATH=. python tools/analyze_scheduler_baselines.py \
  --trace-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime/hr \
  --dataset-name vidore_v3_hr_bm25_colpali \
  --output results/scheduler-baselines/current-hr.json

PYTHONPATH=. python tools/analyze_scheduler_baselines.py \
  --trace-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime/finance \
  --dataset-name vidore_v3_finance_bm25_colpali \
  --output results/scheduler-baselines/current-finance.json
```

The default command evaluates K={10,20,50}, batch={1,4,8,16}, and ten fixed
random seeds.  The same CLI detects the legacy flat trace layout for the two
additional stress-test JSON files.

## Interpretation and next gate

The data now rules out the easiest alternative stories:

- it is not merely FIFO luck or random order;
- it is not reproduced by full-stream popularity;
- it is not reproduced by static high-overlap grouping;
- it is not generally reproduced by shortest-missing or reuse-only;
- the remaining offline work headroom is small at K=20/batch=8.

The next uncertainty is no longer the baseline strength of the queued
full-lookahead result.  It is whether the same advantage survives bounded
arrival windows and whether earlier evidence improves answer-level time to
correctness.  Those experiments should run before further GPU scheduler
replays.
