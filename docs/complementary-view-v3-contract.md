# Complementary-View Optimization V3 Contract

## Paper-level question

ReprForge does not choose an embedding dimension for every page. It decides
which expensive visual page views should physically exist beside a cheap text
index for a known query workload and a fixed construction budget.

Candidate-relative visual evidence is complementary. Materializing one page
alone is not a stable unit of value: at least an incumbent and a challenger
must both exist before their visual scores can be normalized and compared.
The first V3 formulation is therefore

\[
  \max_{|S|\le B}\sum_{(u,v)\in E}w_{uv}\,1[u\in S\land v\in S].
\]

Pages are vertices, boundary comparisons are edges, `B` is the number of
visual views the system may build, and `w` is learned only from other workload
groups. Under a uniform page cost this is weighted densest/heaviest
`k`-subgraph, a monotone supermodular objective. It is not maximum coverage.

This mapping matters because it prevents a false algorithm claim. ReprForge
compares against the diagonal-loading Frank--Wolfe relaxation from
[Lu et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33339)
and labels that solver as prior art. Database systems such as
[Quake](https://www.usenix.org/system/files/osdi25-mohoney.pdf) adapt a vector
index's physical partitions to a workload; V3 instead studies which
heterogeneous document representations should be materialized. The overlap is
the estimate--act--verify pattern, not the optimized object.

## Competing objectives

V3 tests two explicit objectives rather than treating a heuristic as an
algorithm:

1. **Additive edge mass:** the weighted induced-edge objective above.
2. **Query-saturated evidence:** for each query, completed edges contribute
   `1 - product(1 - w_edge)`. Repeated comparisons for one query have
   diminishing value, so the planner cannot spend the whole budget explaining
   one query many times.

The second objective is a hypothesis, not a novelty claim. It is retained only
if its value predicts held-out retrieval better than additive edge mass.

## Implemented solvers and checks

- exhaustive combinations for small-instance ground truth;
- independent incident risk and the existing conditional heuristic;
- a weighted, sparse adaptation of the published diagonal-loading
  Frank--Wolfe baseline;
- deterministic multi-start one-swap search;
- exact, sparse marginal-greedy and multi-start solvers for query-saturated
  evidence. Multi-start is a quality diagnostic; sparse greedy is the
  deployable transfer candidate.

The scalable Frank--Wolfe path stores only observed graph edges. It does not
allocate an `N x N` matrix. Exact enumeration is restricted to registered
18-vertex induced slices.

## Leakage and evaluation boundary

IRPAPERS uses five source-paper-disjoint folds. For each held-out fold:

- the planner may see BM25 candidate identifiers, ranks, margins and page
  reuse;
- flip risks and missing-view priors are fit on the other source-paper folds;
- held-out visual scores and qrels are invisible until execution/evaluation;
- every policy receives the same page count and executor;
- full visual Top-5 is a diagnostic teacher, not a deployable policy.

Report objective value, exact-teacher agreement, Recall@5, selected pages,
solver time and exact-oracle ratio at 10%, 15%, 20% and 25% budgets.

## Decision gates

IRPAPERS is a mechanism set, not sufficient paper evidence: it has 180
single-gold queries, and BM25 already places 78.3% of answers in Top-5.

Proceed to a physical or larger transfer only if a frozen V3 objective:

- is exact or within 1% of exact on all registered small slices;
- is no worse than conditional Recall@5 in at least four of five folds;
- improves at least two held-out queries without using extra pages; and
- adds less than 5% of the measured representation-construction time.

A paper-level claim additionally requires the same unmodified mechanism on
both official ViDoRe HR and Finance-EN (or another graded public transfer): at
least 0.01 absolute nDCG@10 gain at equal measured construction cost, or at
least 20% fewer materialized visual pages at matched nDCG. It must compare with
text-only, full visual, frequency, independent risk, conditional pair
admission and the strongest applicable physical-index baseline.

If transfer fails, V3 is an objective-sufficiency result and not a new system
algorithm. Threshold repair on the transfer datasets is forbidden.
