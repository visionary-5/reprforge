# Pairwise What-If Admission Contract

## Research question

ReprForge can already rank candidate pages independently by how often their
locator rank changes Top-5 membership after visual evidence is available. The
unresolved question is whether that abstraction is sufficient.

A candidate-relative multimodal score is not an independent page property. A
challenger changes Top-5 membership by overtaking an incumbent, and a visual
score normalized over only one newly represented page is undefined. The
candidate mechanism therefore models the boundary workload as a weighted
graph:

```text
page representation = vertex
incumbent--challenger comparison = edge
historical flip risk * locator-margin risk = edge weight
```

The hypothesis is that spending a representation budget on complete,
high-risk comparisons preserves the fully visual Top-5 better than ranking
pages by independent occurrence or incident weight.

This is a falsifiable mechanism hypothesis, not a current contribution claim.

## Public experiment

The first real gate uses the complete public IRPAPERS corpus:

- 3,230 rendered pages;
- 180 questions from 19 source papers;
- BM25 as the cheap locator;
- the official `vidore/colpali-v1.1` adapter as expensive visual evidence;
- BM25 Top-10 as the candidate cohort and Top-5 as the requested result;
- five deterministic, source-paper-disjoint folds.

Training folds may estimate rank-conditioned flip risk and missing-visual
priors. A held-out plan may inspect its BM25 candidates, ranks, scores and
reuse, but not qrels or held-out visual scores. Complete visual scores are
opened only by the replay executor after the admitted page set is frozen.

The full BM25+ColPali Top-5 is a diagnostic teacher, not a deployable oracle.
It receives all visual scores and is used to decompose representation loss.
IRPAPERS qrels are used only for final Recall@5 reporting.

## Same-budget policies

At representation budgets of 20%, 40% and 60% of the unique pages in each
held-out workload episode, compare:

1. **frequency:** pages with the most Top-10 appearances;
2. **boundary independent:** pages with the largest sum of train-only
   rank-flip risks;
3. **pair incident:** pages with the largest sum of incident edge weights;
4. **pair conditional:** greedily seed complete high-value pairs, then prefer
   pages that complete weighted edges next to an admitted anchor;
5. **full visual teacher:** all candidate representations, outside the budget.

All partial policies use the same executor and the same train-only
rank-conditioned prior for missing visual evidence. A policy is not credited
for selecting a page unless its representation is charged.

## Pre-registered decision gate

The pairwise mechanism is retained as a design contribution only if at least
one operating point satisfies both:

- exact Top-5 set agreement improves by at least 3 percentage points over
  both frequency and boundary-independent admission at the same page budget,
  **or** it uses at least 10% fewer selected pages at matched agreement; and
- Recall@5 loses no more than one of 180 queries relative to the stronger
  cheap baseline.

The result must also be directionally consistent in at least four of the five
held-out folds. Pair-edge coverage is a mechanism diagnostic, not a substitute
for retrieval quality.

If the gate fails, ReprForge will not claim pair-aware selection. The failure
would mean that the relational score semantics matter for calibration but do
not justify this admission mechanism. The next design review would then test
a sparse estimator of pairwise score deltas while retaining the stronger
independent admission policy.

## Synthetic preflight

The control plane is tractable, but the heuristic is not uniformly better on
its own synthetic objective:

| Profile | Boundary edges | Page budget | Frequency | Incident | Conditional |
|---|---:|---:|---:|---:|---:|
| IRPAPERS-like | 900 | 121 | 48.29% | 60.75% | **62.25%** |
| ViDoRe-like | 46,500 | 2,039 | 69.71% | **84.31%** | 84.26% |

Percentages are weighted edge coverage only. Conditional planning takes about
11 ms and 480 ms respectively on the local CPU. The large-profile miss is
kept explicitly: a real-quality win cannot be inferred from the graph
objective or from control-plane speed.

## Resource contract

The score-surface restoration uses one explicitly selected idle A100, the
existing read-only PyTorch environment and pinned local model files. No
package installation or environment modification is allowed. The run should
finish within 20 minutes, use no more than one GPU and stop after 30 minutes.
Only compact JSON/NPZ results are copied back into the repository; rendered
pages and model weights remain outside Git.
