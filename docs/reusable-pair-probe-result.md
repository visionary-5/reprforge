# Reusable Pair-Probe Result

## Decision

The reusable-probe implementation is physically sound, but its online
adaptation is not the next ReprForge mechanism.

On the public 3,230-page IRPAPERS benchmark, the pair policies encoded 100
pages per complete five-fold repetition rather than 135 for independent 20%.
The final four-policy matrix measured:

| Policy comparison | repetition 0 | repetition 1 |
|---|---:|---:|
| independent 20% / active pair 15% | 1.135x | 1.197x |
| independent 20% / round-based static pair 15% | 1.199x | 1.234x |
| independent 20% / atomic static pair 15% | **1.228x** | **1.240x** |

The speed is real rather than inferred from page count: query encoding, every
image representation build, all materialization calls and every MaxSim score
were charged. The 40 runs used one A100-SXM4-80GB and alternated policy order
within source-paper folds.

However, the physical visual scores did not preserve the offline adaptive
gain. Active, round-based static and atomic static pair admission all reached
**81.67% Recall@5**; independent 20% reached **80.56%**. Active used 23
materialization rounds per five-fold repetition versus five physical compiler
submissions for atomic static. Active was 3.1--5.7% slower than round-based
static, and missed the 1.15x speed gate in repetition 0. Both frozen active
gates therefore fail.

The system decision is:

> Reject online pair-delta adaptation. Retain static complementary-pair
> admission as the constructive planner and as the next transfer baseline.

## What was built

This iteration removes the old double payment for what-if evidence.

- `ReusablePairProbePlan` selects complete incumbent--challenger comparisons
  under a physical time budget.
- A leakage-checking provider refuses every visual score whose page has not
  been materialized.
- `CohortCompiler.materialize_items` incrementally appends an atomic resident
  generation and never re-encodes an existing page.
- The physical provider encodes each fold's queries, constructs selected page
  views, scores new pages against all fold queries and retains both embeddings
  and scores for later boundary edges.
- The runner uses a deterministic 511-page contract rendered from all 166
  public IRPAPERS PDFs. The benchmark input preparation is separate from the
  charged index build.

The implementation demonstrates that progressive physical design is feasible;
the negative decision concerns the value of the online update, not executor
correctness.

## Offline evidence

Raw pair deltas are informative in isolation. Across 900 Top-5 boundary
events, challenger-minus-incumbent ColPali delta predicts whether the
challenger enters the full-cohort teacher Top-5 with **0.882 AUC**. At the
development 15% page budget:

| policy | pages | Recall@5 | exact teacher agreement | rounds |
|---|---:|---:|---:|---:|
| frequency | 100 | 80.56% | 38.89% | one atomic plan |
| independent risk | 100 | 80.56% | 43.89% | one atomic plan |
| static pair | 100 | 81.67% | 47.78% | 14 |
| active pair | 100 | 82.78% | 47.78% | 23 |

The active point was no worse than the strongest same-budget cheap baseline
in four of five source-paper folds and used 20% fewer pages than the previous
125-page pair controller. But the eight-observation switch was selected on
IRPAPERS and the physical recomputation did not reproduce its extra two
queries. This is development evidence, not a generalization result.

## Mechanism retained

Static pair admission addresses a real abstraction failure. A page is not
valuable independently: at a Top-k boundary, visual evidence is useful only
when enough of the incumbent--challenger comparison is observable. The
planner therefore builds a weighted boundary graph:

```text
query boundary margins
        -> weighted incumbent--challenger edges
        -> select page endpoints that complete high-risk edges
        -> reuse shared pages across edges and queries
        -> atomically compile the selected visual views
```

This is more specific than “use BM25 to pick K pages.” It is a budgeted
complementary-view physical-design problem. Independent per-page risk cannot
express that both endpoints are required or that one resident page may close
several comparisons.

The physical result is a positive systems point: compared with independent
risk, atomic pair composition removes **25.93%** of page builds, improves
Recall@5 by **1.11 percentage points**, and accelerates charged construction by
**1.228--1.240x**. It is not yet a paper-level claim because the policy was
developed and evaluated on one single-gold benchmark and has not been compared
with the strongest optimization formulation.

## V3 research contract

The next version should stop tuning online updates and formalize the static
problem:

> Given query boundary edges, physical page costs and a build budget, choose
> a reusable set of representation views that maximizes the value of complete
> comparisons, then compile that set into a heterogeneous index.

Three pieces are required before a paper claim:

1. **Optimization baselines.** Compare the current greedy pair planner with
   independent utility, frequency/fixed-K, random, a quadratic or integer
   optimization oracle on tractable cohorts, and a scalable relaxation. This
   determines whether the gain comes from the problem formulation or merely a
   heuristic threshold.
2. **Unchanged public transfer.** Freeze the risk function and planner on
   IRPAPERS, then run a graded ViDoRe dataset without threshold repair. Report
   nDCG@10, Recall@5, page builds, vector bytes, build time and query latency.
3. **End-to-end use.** Execute the selected index in retrieval and at least one
   answer/evidence task. A representation plan must preserve agent-usable
   evidence, not only teacher ranking agreement.

If pair composition does not outperform independent admission on the transfer
benchmark under equal measured cost, the mechanism remains a useful IRPAPERS
specialization but should not be the main paper contribution.

## Reproduction and provenance

The compact result is
`results/systems/reusable-pair-probe.json`. It records hashes for the raw
40-run A100 artifact and the complete offline budget sweep. Large rendered
pages, model weights, score surfaces and raw traces stay outside the repository.

Relevant entry points:

- `tools/build_irpapers_candidate_page_contract.py`;
- `tools/analyze_reusable_pair_probe.py`;
- `tools/run_reusable_pair_probe_physical.py`;
- `tools/analyze_reusable_pair_probe_physical.py`.
