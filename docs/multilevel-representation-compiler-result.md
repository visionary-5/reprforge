# Multilevel representation compiler: artifact audit and oracle headroom

Date: 2026-08-05. Decision: **NO-GO-current-artifacts for a deployable dynamic
compiler; GO for one targeted measurement round.**

## Outcome first

The repository contains real three-tier quality surfaces, but it does not yet
contain one closed dynamic physical-design experiment. FinanceEN and Industrial
both align real BM25, pool-4, and full ColPali score matrices with held-out
qrels. They are sufficient for uniform-tier quality and an unattainable
per-query route oracle. They are not sufficient for LRU/LFU/GDSF or a dynamic
physical oracle because the same bundle lacks:

1. a score-comparable query-scoped mixed-document execution surface;
2. the corresponding query-to-item activation stream;
3. per-item compact construction costs;
4. same-workload per-item reload/H2D costs.

MMDocIR supplies the missing physical mechanism and A100 reload lower bound,
but not an arbitrary per-item dynamic quality surface. Combining MMDocIR
latency with ViDoRe quality would be a cross-dataset proxy and is prohibited by
the preregistered contract. Therefore this branch emits explicit `not_run`
records rather than a synthetic dynamic curve.

## What the real headroom says

The query-ID hash split uses two thirds of queries for fit and one third for
evaluation. The best uniform tier is selected on fit qrels. The diagnostic
oracle sees evaluation qrels and chooses one *globally uniform real tier* per
query; it never mixes scores from different tiers inside one ranking.

| Dataset | Eval queries | fit-selected uniform | uniform eval nDCG@10 | query oracle nDCG@10 | oracle gap (95% CI) | oracle choices text / pool4 / full |
|---|---:|---|---:|---:|---:|---:|
| FinanceEN | 109 | text | .5225 | .6330 | +.1105 [.0759, .1492] | 66 / 30 / 13 |
| Industrial | 96 | text | .4470 | .5696 | +.1227 [.0829, .1684] | 54 / 24 / 18 |

All three routes are selected on both held-out workloads. The effect is not a
small full-versus-compressed perturbation: many queries prefer text, while a
substantial minority prefer pool-4 or full. This reproduces the repository's
earlier heterogeneity signal with a genuine third tier and leaves statistically
clear quality headroom.

However, the oracle needs every selected uniform bank eagerly available. Its
union occupies 974,397,824 bytes on Finance and 1,733,985,152 bytes on
Industrial, about 25% more than the full bank alone. It is a quality ceiling,
not a cost win or deployable compiler.

## The important design correction

These states are not a monotone quality/cost ladder:

- Finance eval nDCG@10 is .5225 text, .4396 pool-4, and .4524 full;
- Industrial eval nDCG@10 is .4470 text, .4384 pool-4, and .4504 full;
- on Industrial, producing pool-4 from the existing full bank took a measured
  947,245.6 ms in addition to 476,994.9 ms full visual encoding. The current
  compact representation is cheaper to store and score, but not cheaper to
  construct.

Thus `upgrade` is the wrong universal abstraction. A representation can be
larger yet worse for a query, and compact/full share an expensive encoder
prefix. The database object is a **representation build DAG plus query-scoped
activation**, not an ordered quality ladder. Decoupling residency from
activation remains an enabling mechanism, not the contribution by itself.

The candidate paper-level contribution becomes more specific:

> compile a non-monotone multi-representation index by jointly choosing which
> shared build prefixes and derived views to materialize, while activating only
> the query-beneficial views under storage, build, and reload budgets.

This is still a hypothesis. It is justified for further measurement by the
three-way oracle choices, but not yet supported as a systems result.

## Prior branches that are now closed

- Binary retain/evict control is not the method: GDSF/no-cache closes that
  comparison under the existing two-state contract.
- Pool-4 plus static full anchors is not the method: its certificate was
  false-safe on both Industrial and Pharmaceuticals (nDCG upper regret .01086
  and .01104).
- Ordinary residual CUR/SVD completion is not the method: residual energy is
  not low-rank enough and ranking fidelity did not protect relevance.
- A compact state must not be assigned a fake cheap build cost: the Industrial
  measurement shows the opposite with the present implementation.

## Strong-baseline status

Uniform text, pool-4, full, and full-eager are evaluated from real held-out
surfaces. Static content rule, LRU, LFU, GDSF, transient refinement, and the
oracle physical design are all `not_run_missing_artifact` in both JSON reports.
They need the same mixed-state outcome, activation trace, capacity, and
transition costs; running cache policies on a quality-independent request list
would merely repeat the closed binary-retention experiment.

## Smallest next experiment

Do not implement a learned controller yet. Extend the existing MMDocIR
query-scoped pool-25/full execution to persist one unified artifact:

1. locator candidate/activation IDs for every query step;
2. exact pool-only and each query-scoped full-activation outcome needed to
   evaluate an arbitrary retained subset without global score pollution;
3. per-item full encode time, compact/full bytes, and batched cold/warm H2D
   reload measurements on the same A100;
4. representation lineage: full encoding, compact derivation, retained parent
   tokens, and the cost of recreating a discarded parent;
5. at least two frozen MMDocIR document roles or workload streams.

Then run the registered static and dynamic oracles before any online method.
Continue only if an oracle saves at least 10% charged build+reload+storage cost
against the strongest full-eager, transient, and GDSF baseline at matched
nDCG@10 on both roles. A failed oracle would close the compiler direction
cheaply; a passed oracle would tell us exactly which planner action matters.

## Reproduction and artifacts

The deterministic analyzer is
`tools/analyze_multilevel_physical_design.py`. Raw server score files are not
committed; their SHA-256 digests and absolute replay paths are recorded in:

- `results/multilevel-representation-compiler/finance.json`;
- `results/multilevel-representation-compiler/industrial.json`.

The copied pool-4 digests match the previously frozen compression-risk and
physical-plan artifacts. The result JSONs record every cost source and preserve
missing reload values as `null`.
