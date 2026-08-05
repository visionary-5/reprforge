# Multilevel representation compiler: preregistered feasibility and headroom contract

Date: 2026-08-05. Branch: `exp/multilevel-representation-compiler`.
This contract was written before copying or evaluating the server-side pool-4
score surfaces for this experiment.

## Question

Can ReprForge treat text, a compact visual multi-vector representation, and a
full visual multi-vector representation as physical database states, and can a
planner improve the quality--work--storage--latency frontier by choosing when
to stay, upgrade, retain, reload, or evict each state?

This branch is a headroom and artifact-feasibility test, not a new controller.
It must not turn representation decoupling itself into the claimed
contribution. A controller is justified only if a real multistate oracle leaves
material headroom over strong physical-design and caching baselines.

## Frozen evidence boundary

The following earlier outcomes are treated as closed evidence and will not be
rerun as new contributions:

- binary full-vector retention is closed by the strong GDSF/no-cache result;
- pool-4 plus static full anchors is closed by two false-safe validation
  collections;
- ordinary residual CUR/SVD completion is closed by the measured residual
  spectra and cross-domain quality failures;
- pair-aware admission has a real low-budget quality effect but did not reduce
  physical builds enough to pass its speed gate;
- MMDocIR establishes a real pool-25 base plus full visual delta and measured
  query-scoped activation, but its current content rule is static and its
  public result does not expose a dynamic per-item quality surface.

## Representation states and mandatory measurements

The intended states are:

1. `cheap_base`: BM25/text scores, per-item build cost and storage;
2. `compact_pool`: pool-4 or pool-25 visual scores, per-item storage and a
   measured construction path;
3. `full_multivector`: full visual scores, per-item build cost and storage.

A dynamic three-state replay may run only when one workload supplies all of:

- aligned query and item IDs for all three real score surfaces;
- held-out qrels or a document/workload split;
- a real, score-comparable mixed-state execution rule;
- per-item build costs for both visual transitions;
- per-item persistent bytes for every state;
- per-item reload or host-to-device costs measured on the same representation
  and hardware;
- a query stream exposing which items each query may activate.

Aggregate bank build time is not a per-item reload profile. A measurement from
MMDocIR may not be silently transplanted to ViDoRe. Missing values remain
missing; the implementation must never synthesize an intermediate quality or
latency surface.

## Registered experiment order

1. Audit the existing artifacts and emit a deterministic JSON capability
   matrix with hashes, present fields, blockers, and runnable analyses.
2. If the complete three-state surface exists, run a static physical-layout
   oracle and then a dynamic oracle before implementing an online method.
3. Otherwise, run only the strongest honest fallback: a held-out uniform-tier
   and per-query route headroom analysis over every aligned real tier. This is
   an oracle diagnostic and is not a deployable physical planner.
4. Do not execute dynamic baselines when their common cost/quality contract is
   incomplete. Emit one structured `not_run` record per baseline instead.

The fallback query split is the existing deterministic query-ID hash split:
two thirds are `fit`, one third is `eval`. Uniform-tier selection uses fit
qrels and is evaluated once on eval. The per-query route oracle uses eval qrels
and is explicitly labelled unattainable. Primary quality is nDCG@10; Recall@100
is secondary. Storage is the sum of actual vector bytes of the materialized
tiers. Build work is reported only when the archive contains a measured field.
No cross-tier mixed score is constructed in the fallback.

## Strong baseline registry

The complete dynamic experiment must compare:

- uniform cheap, compact, and full tiers;
- a static content/type rule;
- LRU, online LFU, and GDSF under the same byte budget;
- full eager materialization;
- transient refinement with no retention;
- a clairvoyant physical-design oracle.

Every baseline must share the exact same logical retrieval surface, stream,
capacity, and cost vectors. If that is impossible, its status is
`not_run_missing_artifact`; it must not receive a proxy result from another
dataset.

## Frozen decisions

The branch may recommend building a controller only if:

1. static or dynamic oracle saves at least 10% total charged cost relative to
   the strongest uniform/GDSF/transient baseline at matched nDCG@10 (absolute
   regret at most 0.01);
2. the effect holds on two held-out workloads or document roles;
3. the comparison includes storage, build work, reload work, and measured or
   faithfully replayed latency without mixing hardware/data provenance.

If three-state feasibility fails, the decision is `NO-GO-current-artifacts`,
not a claim that multilevel physical design is impossible. The result must name
the smallest next measurement that would unlock the experiment.
