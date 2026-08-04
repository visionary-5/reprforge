# Bounded-arrival cohort-frontier replay

Date: 2026-08-04. Branch: `exp/windowed-arrivals`.

## Decision

**NO-GO under the pre-registered W=16/W=32 gate.** On the current frozen HR
and Finance traces, W=32 improves completion work and quality--work AUC in all
four domain/arrival settings, but preserves only 49.5% (HR) and 48.7% (Finance)
of the full-pending gain under burst arrivals. Both are just below the required
50%. W=16 fails more clearly.

This is not a negative result for the whole mechanism. W=64 passes the 50%
completion-gain criterion in every setting, and every tested frontier window
has higher AUC than FIFO. The defensible conclusion is narrower: **the current
frontier advantage survives causal arrivals when about 64 pending requests are
visible; the evidence does not yet support the stronger “16--32 requests are
enough” claim.**

There is also a tail trade-off. Unconstrained frontier improves mean completion
work but can repeatedly bypass old requests. A hard fairness cap removes the
measured starvation, but no longer preserves half of the full-pending gain.

## What was tested

Each query first exposes only its cheap BM25 top-20 candidate pages. At every
atomic construction batch, a scheduler can inspect only the oldest W arrived
and pending queries. It can observe their candidate-page membership and the
pages already resident or staged. It cannot inspect future arrivals, relevance
labels, visual scores, or per-query quality gain.

The comparison includes:

- FIFO;
- deterministic random selection inside the visible window;
- a causal history-popularity baseline whose priority is frozen at arrival
  from earlier arrivals only;
- bounded cohort frontier;
- an additional hard-fair frontier that forces a query before W younger
  arrivals can bypass it.

Windows are 1, 8, 16, 32, 64, plus a full-pending diagnostic. Results aggregate
five fixed shuffled query streams (seeds 20260804--20260808). Burst arrivals use
32 queries every 64 encoded-page work units. Poisson arrivals use a mean
interarrival of four encoded-page work units.

`completion work` is the number of unique visual pages resident when an atomic
query batch is published. `quality--work AUC` integrates frozen mean nDCG@10
over encoded-page work. Labels are used only after a scheduler decision to
measure that curve.

## Frozen data provenance

| Trace | Queries | Corpus pages | Top-20 union | BM25 / fused nDCG@10 |
|---|---:|---:|---:|---:|
| Current HR | 318 | 1,110 | **895** | 0.48600 / 0.53729 |
| Current Finance | 309 | 2,942 | **1,855** | 0.52852 / 0.56280 |

These unions exactly match the current candidate-compiler experiments. The
loader verifies the trace manifests and runtime/label SHA-256 digests. HR BM25
and visual runtime digests are `f057d766...a1bc84` and
`8a6e8ec2...c7eef`; Finance digests are `55bf5d99...7fa65` and
`a6bbbf5b...35944`. The output JSON records full digests, absolute source paths,
file sizes, manifest digests, and the upstream ViDoRe revision.

## W=32 result

Values combine all five deterministic permutations. Wait is measured in page-
work clock units; it is not GPU wall time.

| Data / arrival | Policy | Mean / P50 / P95 completion pages | Quality--work AUC | P95 / max wait | Starved fraction |
|---|---|---:|---:|---:|---:|
| HR / burst | FIFO | 743.1 / 823 / 891 | 0.50339 | 574 / 603 | 0% |
|  | Random | 746.1 / 823 / 892 | 0.50322 | 621 / 819 | 15.2% |
|  | History popularity | 733.8 / 811 / 893 | 0.50454 | 887 / 895 | 9.9% |
|  | **Frontier** | **664.3 / 740 / 854** | **0.50738** | 662 / 870 | 19.8% |
|  | Hard-fair frontier | 708.0 / 801 / 885 | 0.50535 | 573 / 617 | **0%** |
| HR / Poisson | FIFO | 742.4 / 823 / 891 | 0.50338 | 387 / 416 | 0% |
|  | Random | 741.9 / 821 / 891 | 0.50356 | 454 / 672 | 10.0% |
|  | History popularity | 724.4 / 806 / 891 | 0.50418 | 718 / 851 | 10.1% |
|  | **Frontier** | **690.3 / 747 / 891** | **0.50622** | 492 / 803 | 13.3% |
|  | Hard-fair frontier | 713.7 / 799 / 891 | 0.50506 | 405 / 468 | **0%** |
| Finance / burst | FIFO | 1,272.9 / 1,404 / 1,841 | 0.54797 | 1,265 / 1,279 | 0% |
|  | Random | 1,274.4 / 1,402 / 1,836 | 0.54776 | 1,326 / 1,708 | 15.1% |
|  | History popularity | 1,202.3 / 1,303 / 1,805 | 0.54869 | 1,726 / 1,849 | 11.7% |
|  | **Frontier** | **1,086.4 / 1,163 / 1,698** | **0.55074** | 1,363 / 1,720 | 20.1% |
|  | Hard-fair frontier | 1,187.0 / 1,304 / 1,772 | 0.54886 | 1,247 / 1,310 | **0%** |
| Finance / Poisson | FIFO | 1,272.1 / 1,403 / 1,834 | 0.54796 | 833 / 860 | 0% |
|  | Random | 1,272.2 / 1,398 / 1,831 | 0.54781 | 956 / 1,424 | 14.8% |
|  | History popularity | 1,178.2 / 1,269 / 1,779 | 0.54946 | 1,600 / 1,818 | 12.4% |
|  | **Frontier** | **1,087.7 / 1,152 / 1,693** | **0.55060** | 1,081 / 1,645 | 20.0% |
|  | Hard-fair frontier | 1,188.4 / 1,300 / 1,775 | 0.54891 | 842 / 903 | **0%** |

The starvation definition is observable and strict: a query is counted when
at least W younger arrivals are dispatched first. At W=32 the unconstrained
frontier's maximum younger-bypass count reaches 226--307, so the tail issue is
not a rounding artifact.

## Window gate

| Data / arrival | W=16 gain preserved | W=32 gain preserved | W=64 gain preserved | W=32 AUC delta vs FIFO |
|---|---:|---:|---:|---:|
| HR / burst | 29.1% | **49.5%** | 72.9% | +0.00400 |
| HR / Poisson | 54.7% | **83.9%** | 100.0% | +0.00284 |
| Finance / burst | 30.5% | **48.7%** | 73.9% | +0.00277 |
| Finance / Poisson | 38.0% | **62.0%** | 90.1% | +0.00264 |

W=32 reduces mean completion work versus FIFO by 10.6% / 7.0% on HR and
14.7% / 14.5% on Finance for burst / Poisson arrivals. It beats random and
causal history popularity on both completion work and AUC. Nevertheless, the
pre-registered rule is conjunctive, so the two burst near-misses make the
formal decision NO-GO. No threshold or arrival parameter was changed after
seeing this result.

The hard-fair W=32 variant preserves only 22.0%, 46.2%, 22.4%, and 28.2% of
full-pending gain respectively. It removes starvation, but fails the efficiency
gate by a wide margin.

## Interpretation and next design

The online result is real but conditional on queue depth. A burst of 32 requests
does not quite expose enough cohort overlap to recover half of a full pending
queue's opportunity. A 64-request window does. This gives the paper a useful
systems boundary rather than a universal claim: ReprForge works best for batch
onboarding, concurrent tenants, or queued analytical workloads, and its benefit
shrinks for lightly loaded interactive traffic.

The next scheduler should make the efficiency/fairness trade-off explicit.
Plausible qrel-free designs are an age penalty, a page-work deadline, or two
queues that reserve a fixed fraction of construction batches for oldest-first
service. They should be frozen on one domain or synthetic traces and evaluated
on the other domain; the hard cap tested here is too blunt.

This replay does not justify another GPU run by itself: page-work behavior is
already known, and the W=32 gate failed. A real run becomes useful after a
fairness-aware rule passes frozen replay or if the paper deliberately adopts
W=64/batch-workload scope.

## Reproduction

No SSH, GPU execution, or download is used.

```bash
python -m tools.analyze_windowed_arrivals \
  --data-root /Users/aura/gpu-systems-incubator/reprforge/data/current-anytime \
  --output results/systems/windowed-arrivals-v1.json

pytest -q
```

Machine-readable output: `results/systems/windowed-arrivals-v1.json`.
