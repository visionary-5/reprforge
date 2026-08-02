# Pairwise Representation Admission Result

## Outcome

The first public A100 experiment supports a narrow admission mechanism, but
does **not** yet pass the end-to-end speed gate:

> Under a tight representation budget, visual document views should be
> admitted as complete ranking-boundary comparisons, rather than as
> independently valuable pages.

The relational quality effect survives source-paper-held-out evaluation. A
train-only controller also removes some physical builds. The reduction is not
large or stable enough to deliver the preregistered 1.15x speedup.

## Why the mechanism exists

The previous boundary-weighted allocator treated every page as an independent
database view. That is incomplete for candidate-relative retrieval. A
challenger affects Top-5 only by crossing an incumbent, and the visual score of
one isolated candidate cannot establish the comparison scale used by the
executor.

ReprForge therefore compiles a query workload into a boundary graph:

```text
page representation               -> vertex
rank-5 incumbent vs tail candidate -> edge
flip risk * locator-margin risk    -> edge weight
```

Historical source-paper-disjoint folds estimate rank flip risk without qrels.
For a new episode, the compiler sees only BM25 candidates, scores, ranks and
reuse. Its conditional policy seeds a complete high-value pair and then favors
pages that complete several weighted comparisons next to a selected anchor.
The selected vertices become physical ColPali representations; missing pages
remain logical BM25 candidates and receive a train-only rank prior.

This is the database connection: a representation is a candidate physical
view, boundary edges are the what-if workload, and materialization is charged
only after admission. Unlike an ordinary additive knapsack, the benefit of one
view can depend on another view being present.

## Public experiment

The run uses the pinned public [IRPAPERS corpus](https://github.com/weaviate/IRPAPERS)
and official [ColPali-v1.1 adapter](https://huggingface.co/vidore/colpali-v1.1)
on one NVIDIA A100-SXM4-80GB:

- 3,230 pages from 166 papers;
- 180 questions from 19 source papers;
- five deterministic source-paper-disjoint folds;
- BM25 Top-10 candidates and a Top-5 result boundary;
- complete score logs used only on training folds and by the replay teacher;
- qrels hidden from admission.

The restored dataset SHA-256 is
`8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b`.
The full score surface contains 180 x 3,230 BM25 and ColPali scores.

### Baseline coordinate

| Execution | Recall@5 | Visual pages | Measured time |
|---|---:|---:|---:|
| BM25 only | 78.33% | 0 | 3.36 s build + score |
| Full ColPali | 67.22% | 3,230 | 748.71 s visual build; 759.11 s total |
| Full BM25 + ColPali fusion | 83.33% | 3,230 | reuses the full visual index |
| BM25 Top-10 + visual replay | 82.22% | 511 unique candidates | replay-only |

These rows establish the local cost/quality surface. They do not claim a
controlled speed comparison with the current closed IRPAPERS leaderboard.

## Same-budget gate

At 20% of the eligible episode pages, every partial method selects 135 pages
when summed across the five independent held-out episodes.

| Policy | Exact teacher Top-5 agreement | Recall@5 | Visual score events |
|---|---:|---:|---:|
| Frequency | 50.56% | 81.67% | 1,003 |
| Boundary independent | 50.56% | 81.67% | 943 |
| **Pair conditional** | **56.11%** | **81.67%** | **816** |

Pair conditional improves agreement by 5.56 percentage points over both cheap
baselines with no Recall@5 loss. Relative to the stronger cheap baseline in
each fold, it is strictly better in three folds and tied in two; it never
regresses. This passes the preregistered low-budget mechanism gate.

The effect is budget-local. At 40% and 60%, boundary-independent admission has
higher agreement (68.33% and 80.00%) than pair conditional (66.11% and
76.11%). ReprForge must not claim that relational admission universally wins.

## Physical design review

The same-budget physical run encodes 135 pages for either policy. Pairwise
reduces MaxSim pairs by 13.5%, but total execution is 47.01 s versus 44.31 s.
Visual encoding dominates, and different batch shapes add variance. This
closes a tempting but wrong claim: fewer scoring events alone are not the
end-to-end contribution.

A post-gate budget diagnostic finds a matched-quality point:

| Physical policy | Episode pages | Score pairs | Agreement | Recall@5 | Execution |
|---|---:|---:|---:|---:|---:|
| Boundary independent, 20% | 135 | 943 | 50.56% | 81.67% | 40.05 s |
| **Pair conditional, 17%** | **114** | **735** | **51.67%** | **82.22%** | **32.47 s** |

The pairwise plan builds 15.56% fewer pages and executes 1.23x faster while
slightly improving both reported quality coordinates. Every fold is faster.
However, the 17% budget was selected after reading this dataset's budget
curve, so this establishes design feasibility only.

### Train-only calibration

The deployable controller instead matches the 20% independent policy's exact
teacher agreement using only the other four source-paper folds. It chooses
pair budgets of 14%, 16%, 20%, 20% and 20% for the five held-out episodes.

| Train-only physical policy | Episode pages | Score pairs | Agreement | Recall@5 |
|---|---:|---:|---:|---:|
| Boundary independent | 135 | 943 | 50.56% | 81.67% |
| **Pair conditional** | **125** | **743** | **52.22%** | **82.22%** |

The controller removes 7.41% of physical page builds and 21.21% of score
pairs without consulting held-out qrels or visual scores. That validates the
quality side of the intended causal chain:

```text
conditional boundary coverage
    -> fewer admitted physical views at matched quality
    -> fewer visual encodes
```

The first lazy executor still fragmented those views across 20 encoder calls,
versus 18 for the baseline, and produced unstable timing. ReprForge therefore
added eager atomic materialization: the known admission plan is encoded before
queries and query-time logical activation remains candidate-scoped.

Two fold-interleaved repetitions alternate policy order. Their complete
construction-plus-query speedups are 1.121x and 1.032x; combined speedup is
1.074x. Seven of ten paired fold measurements favor pairwise. Both repetitions
miss the preregistered 1.15x gate. The selected-page reduction occurs in only
two folds, so the current exact-agreement controller is too conservative to
produce a material and stable systems gain.

## What this establishes

1. Candidate representation value can be non-additive in a real retrieval
   consumer; an independent page score leaves measurable low-budget policy
   value.
2. The pair graph is not merely explanatory. Its selected pages can be
   physically encoded by the existing executor and reproduce replay quality.
3. The system bottleneck is page construction, not MaxSim event count. A
   useful allocator must remove substantially more physical builds at matched
   risk; 7.4% is insufficient for the 1.15x target.
4. Workload compilation and materialization scheduling must be separate.
   Eager atomic publication removes query-batch fragmentation but cannot make
   a conservative admission plan more selective.

## What remains before a paper claim

1. Replace exact teacher matching with a train-only risk constraint that may
   spend an explicit quality-loss allowance and is judged by held-out qrels.
2. Avoid requiring a complete historical visual index. Measure sparse shadow
   probes and an estimate--then--verify loop.
3. Transfer the mechanism to a graded public ViDoRe workload. One single-gold
   dataset cannot establish generality.
4. Compare against fixed K, frequency, independent boundary risk, random and a
   stronger learned/additive allocator under equal physical-build cost.
5. Add a query stream and representation invalidation so that maintenance and
   workload drift are charged rather than discussed.

The current contribution candidate is therefore not “BM25 picks some pages”
or “use a smaller K.” It is a workload compiler for complementary physical
representations, with the unresolved research problem concentrated in sparse
what-if estimation and risk-constrained budget control. The pair selector is
retained; the current physical performance claim is rejected.

## Artifacts

- `reprforge/pairwise_view_admission.py`: deterministic boundary-graph builder
  and conditional admission heuristic;
- `reprforge/pairwise_budget.py`: qrel-free train-only budget calibration;
- `tools/analyze_pairwise_view_admission.py`: qrel-blind held-out score replay;
- `tools/run_pairwise_admission_physical.py`: cold A100 execution of admitted
  representations;
- `results/systems/pairwise-view-admission-result.json`: compact result and
  immutable source hashes.
