# Progressive Visual Index Contract

> **Status update (2026-08-01).** The independent page-utility branch defined
> below has been executed and rejected: cohort interactions dominate the
> single-page labels, and held-out prediction fails. The document remains the
> preregistration record. The active mechanism is candidate-relative fusion
> followed by asynchronous cohort compilation; see
> `candidate-relative-fusion-result.md`.

## Research question

Can a multimodal retrieval system begin with a cheap corpus-wide
representation and progressively build only the expensive visual state that a
real query stream proves useful?

The question is not whether query-driven indexing exists. The rejected
`tiered-selective K=20` policy already performs its simplest form: a query
touches pages, missing visual embeddings are built, and every built embedding
is retained. It reaches 932 of 1,110 pages because **candidate membership is
incorrectly treated as permanent admission evidence**.

The system problem is therefore to separate three decisions that K=20
conflates:

1. locate a page cheaply;
2. refine it visually for the current query;
3. admit the resulting representation into a persistent visual index only
   when expected future benefit amortizes build, search, and storage cost.

Eviction is the symmetric decision under workload drift or a fixed cache
budget.

## Position relative to current work

Database cracking and progressive indexing supply the query-driven physical
design pattern, not a ready-made policy for retrieval quality.
[Quake](https://arxiv.org/abs/2506.03437) supplies an especially useful
maintenance workflow: track observable statistics, estimate an action's cost
delta, verify promising actions, then commit or reject them. Its cost model
predicts ANN partition scan latency; ReprForge must instead estimate a
ranking-quality benefit whose labels are sparse and whose actions change
competing scores.

Two 2026 visual-retrieval systems establish strong adjacent baselines:

- [MURE](https://arxiv.org/abs/2603.13349) builds a compact multi-resolution
  visual representation for every page. Its 512-token result shows that a
  fixed compact visual tier may be much stronger than full ColPali. It does
  not allocate representation across pages from an online workload.
- [LightSTAR](https://arxiv.org/abs/2606.23539) applies a lightweight visual
  selector corpus-wide and expensive MLLM refinement only to query candidates.
  It directly covers transient selective refinement and reports 2.3x lower
  latency than ColPali at 7,000 pages. Its
  [official repository](https://github.com/bokufa/LightSTAR) currently says
  code and weights are forthcoming. ReprForge must compare against this
  execution model; persistence is useful only if cross-query reuse beats
  repeated transient refinement under the same quality target.

The proposed research axis is orthogonal to both: **which page
representations should become persistent physical index state as the workload
evolves?** A final evaluation should combine the lifecycle policy with a
current compact visual representation when a reproducible artifact is
available.

## State and actions

Every page always has a cheap base representation. Its visual state is one of:

- `absent`: no expensive representation exists;
- `transient`: encoded for the current query, then discarded;
- `resident`: encoded and searchable for future queries;
- later, optionally, one of several resident resolution/token tiers.

For query episode `t`, the controller may:

- `stay`: use the base representation;
- `refine`: build visual state for the current query only;
- `admit`: retain a verified transient representation;
- `retain`: keep resident state;
- `evict`: remove resident state when its expected value decays.

The first implementation must use only `absent/transient/resident`. A
multi-resolution ladder is deferred until this binary lifecycle has measured
headroom.

## What-if value and causality

The offline diagnostic may use qrels and both score matrices. For a resident
set `S`, let `Metric(S)` be official ViDoRe nDCG under frozen mixed-index
semantics. The exact marginal value of admitting page `i` is

\[
\Delta U_i(S) = Metric(S \cup \{i\}) - Metric(S).
\]

This value is not assumed additive. Ranking competitors can create pairwise
interactions:

\[
\Omega_{ij}(S) = \Delta U_{i,j}(S)-\Delta U_i(S)-\Delta U_j(S).
\]

The finite-difference form is inspired by forward sensitivity analysis such
as [CLADO](https://arxiv.org/abs/2307.05657), but no quantization theory,
differentiability, or Hessian claim is transferred. CLADO uses labelled
calibration loss and pairwise layer perturbations; here qrels are permitted
only for an offline oracle or a training split.

A deployable policy must not predict relevance from test qrels. It follows an
estimate--verify--commit loop:

1. estimate reuse and possible value from pre-visual signals;
2. transiently encode only promising current-query candidates;
3. observe the actual score/rank intervention for the current query;
4. admit only when observed evidence and predicted future reuse amortize cost;
5. evict under the same budgeted value model.

This narrows the hard prediction problem. The policy predicts **future
amortized reuse of a measured intervention**, not the unseen visual embedding
quality from layout heuristics alone.

## Frozen trace required before policy code

The next A100 run must export a compact, model-independent trace containing:

- ordered query and corpus identifiers;
- every query--page Markdown score;
- every query--page full-visual score;
- qrels, separated from runtime-visible fields;
- per-page text and visual vector bytes;
- per-page visual encode time, including batching metadata;
- query encode and score time;
- source/model/data hashes and the exact official evaluator revision.

The 318 by 1,110 score matrices contain only 352,980 values each and are
small enough to replay locally. Raw images and embeddings remain outside Git.
A sanitized replay artifact may contain scores, costs, identifiers, and qrels
if the upstream licences permit redistribution.

The replay engine must define score calibration and fusion explicitly. Raw
replacement, calibrated score fusion, and rank fusion are distinct baselines;
the oracle may not silently choose among them per query.

## First headroom experiment

Run a static end-state oracle before simulating a dynamic workload. This
answers whether any set of at most 30% resident pages can close the measured
quality gap under one fixed, deployable fusion rule.

The oracle succeeds only if it simultaneously:

- admits at most 333 of 1,110 pages;
- reaches at least
  `text + 0.95 * (full_visual - text) = 0.51662` nDCG@10 on the frozen HR run;
- keeps base plus resident visual bytes below the full-visual index;
- remains better than frequency-only and relevance-frequency upper bounds
  that ignore ranking interactions.

Use exact enumeration on small slices to validate a greedy marginal oracle.
On the full corpus, report the complete quality--resident-fraction curve and
pair-interaction diagnostics; do not report only the 30% operating point.

If this oracle fails, no online cracking controller can meet the target with
the current base/visual representations and fusion semantics. Change the
locator, visual representation, or fusion rule before policy learning.

## Online replay after oracle success

ViDoRe has relevance judgements but no natural query timestamps. Therefore
its query order is not a production trace. Replay must report separately:

- the official deterministic order;
- multiple frozen random permutations;
- explicitly synthetic clustered episodes and distribution shifts.

The paper may claim workload robustness, not natural temporal realism, until
a public timestamped query stream is added.

Compare:

1. base-only;
2. full visual built before queries;
3. LightSTAR-style transient candidate refinement with no persistence;
4. unconditional admit-on-first-touch (the rejected K=20 rule);
5. LRU, LFU, two-hit admission, and fixed-frequency thresholds;
6. the proposed estimate--verify--commit policy;
7. an offline future-aware oracle, clearly labelled undeployable.

Report official retrieval metrics, cumulative GPU encoding work, query
P50/P95, time to quality, resident bytes, admission/eviction churn, and total
cost at every query horizon. A positive result requires a stable Pareto
improvement over both transient refinement and simple admission/cache rules,
not merely fewer visual pages than full indexing.

## Stop and continuation rules

- Stop policy work if the 30% static oracle misses the quality target.
- Stop learned utility work if LFU/two-hit admission is within 5% of the
  future-aware oracle's end-to-end cost at the same quality.
- Continue to a system paper only if persistence beats transient refinement
  on at least two public workload families or on one public benchmark plus a
  defensible public temporal trace.
- Treat MURE/LightSTAR implementation availability as artifact status, not a
  paper claim. Recheck their official repositories before final evaluation.
