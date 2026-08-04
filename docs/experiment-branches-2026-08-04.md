# ReprForge parallel experiment branches

Date: 2026-08-04. Base branch: `research/anytime-index-base`.

Round-one results are summarized in
`docs/parallel-experiment-round1-result-2026-08-04.md`.  Frozen commits are:
`exp/windowed-arrivals` at `da021d6`, `exp/scheduler-baselines` at `a8009ca`,
`exp/revision-safety` at `e439efc`, and `exp/benchmark-transfer` at `9e268dd`.
The two round-two branches and their frozen gates are registered in
`docs/experiment-round2-contract-2026-08-04.md`.
Round-two results are summarized in
`docs/parallel-experiment-round2-result-2026-08-04.md`: aging-aware frontier
`2b2f495`, value-aware frontier `afb1c54`, and anytime metrics `e67bfc3`.
The third-round and paper-audit branches are:

- `exp/answer-time-to-correct` at `1fc5b43`;
- `exp/capacity-warmstart` at `8a5bcae`;
- `exp/structured-citation` at `20336f7`;
- `exp/answer-semantic-judge` at `e2cb937`;
- `exp/modern-retriever-transfer` at `972ceef`;
- `exp/frontier-scale` at `5cca66f`;
- `exp/vidore-domain-matrix` at `22780c1`;
- `research/closest-work-audit` at `56003ee`.

Their combined result is summarized in
`docs/parallel-experiment-round3-result-2026-08-04.md`.  The two remaining P0
novelty gates are `exp/edgerag-faithful-baseline` and
`exp/cagr-faithful-baseline`; neither has permission to weaken its gate after
seeing HR/Finance results.

The subsequent P0 and method-stress branches are frozen at:

- `exp/edgerag-faithful-baseline` `8f5b21a` (**CONTINUE**);
- `exp/cagr-faithful-baseline` `a3cd197` (conditional page-work GO);
- `exp/cagr-strong-adaptation` `7fd795f` (no deployable HR selection);
- `exp/cagr-bounded-wait` `af893b8` (**STOP/DOWNGRADE** counterexample);
- `exp/time-aligned-quality` `887eaa2` (system--quality trade-off);
- `exp/cost-locality-frontier` `3e2d33c` (NO-GO);
- `exp/frontier-constrained-locality` `225be9a` (NO-DEPLOYABLE);
- `exp/deadline-constrained-locality` `b3b4662` (NO-GO and stop tweaks).

The final decision and corrected claim boundary are summarized in
`docs/paper-decision-after-p0-2026-08-04.md`.

This registry keeps the experiments independent.  A branch may produce a
negative result; it must not change the common K=20 fusion semantics or tune on
held-out qrels merely to manufacture a positive result.  Frozen score replay is
required before any new A100 execution.

## `exp/windowed-arrivals`

Question: does the cohort-frontier advantage survive when the scheduler sees
only a bounded pending window rather than the full benchmark stream?

Required comparisons:

- FIFO, random, static history popularity and bounded frontier;
- windows 1, 8, 16, 32 and 64;
- at least five deterministic query permutations and burst/Poisson arrival
  models;
- mean/P50/P95 completion work, quality--work AUC, starvation and maximum wait.

Go signal: W=16 or W=32 preserves at least half of the full-lookahead completion
work gain on both HR and Finance without worse quality--work AUC than FIFO.

## `exp/scheduler-baselines`

Question: is the current mechanism better than strong, simpler explanations of
the gain?

Required comparisons:

- FIFO and random;
- future-aware static popularity, explicitly marked as an offline upper
  baseline;
- CaGR-style cohort-overlap grouping;
- shortest-missing-cohort first;
- reuse-only scheduling;
- an offline qrel-free greedy construction-work oracle when tractable;
- K in {10, 20, 50}, request batch in {1, 4, 8, 16}.

Go signal: frontier is Pareto-competitive across both domains and its advantage
cannot be reproduced by overlap-only or popularity-only scheduling.

## `exp/revision-safety`

Question: can a label-free publisher reduce harmful BM25-to-visual revisions
without discarding most useful improvements?

Candidate observables are restricted to text/visual rank agreement, top-score
margin, candidate overlap, reciprocal-rank movement and score normalization
statistics.  Qrels are used only for post-hoc evaluation under strict
query-level cross-fitting.

Required comparisons:

- publish every completed visual cohort;
- never revise;
- fixed agreement/margin thresholds;
- cross-fitted label-free feature model;
- an oracle publisher to measure available headroom.

Report improvement retained, harmful-revision precision/recall, abstention,
mean nDCG, fifth-percentile revision loss and worst-5% CVaR.

Go signal: on held-out folds in both HR and Finance, harmful revision frequency
falls by at least 30% while retaining at least 80% of positive nDCG gain and
without reducing mean quality below BM25.

## `exp/benchmark-transfer`

Question: do workload overlap and progressive construction benefits transfer
beyond the two opened ViDoRe domains?

Start with existing local traces and benchmark artifacts before downloading or
encoding new data.  Candidate targets are Computer Science / other ViDoRe v3
domains, IRPAPERS and MMDocIR; report where the task definition differs from
full-corpus retrieval.  MIRACL-VISION or M3DocVQA is the preferred later public
scale test if local artifacts are insufficient.

Required outputs:

- corpus/query/cohort overlap statistics;
- FIFO, popularity and frontier exact-work replay;
- locator coverage and pure-visual-query stratification where labels permit;
- an explicit decision on whether a real GPU run is warranted.

Go signal: at least one independent full-corpus benchmark shows a frontier
completion-work gain with final semantic parity, without dataset-specific
scheduler tuning.

## Shared reporting contract

Every branch must leave:

1. a test-covered implementation;
2. a deterministic command-line replay;
3. machine-readable JSON results;
4. a short Markdown result with a GO / NO-GO decision;
5. exact data provenance and a statement of what the scheduler was allowed to
   observe.

No branch is merged merely because it has a positive mean.  It must beat its
strongest relevant baseline, report tails, and survive at least one transfer or
held-out split.

## Final headroom branch

### `exp/multiobjective-oracle-headroom`

Question: after the simple scheduler variants fail, does a favorable finite
oracle family still contain a policy that simultaneously improves system cost,
elapsed evidence quality, tail latency, and starvation?

The preregistered family contains 60 greedy configurations over qrel-derived
quality density, exact next-step completion cost, deadline pressure, and a
bounded future-arrival wait.  HR is the only selection domain; Finance remains
sealed unless HR yields a safety-qualified point.

Result: no registered configuration is HR-safe.  Twenty-four configurations
pass the primary endpoint and P95 checks, and four pass the starvation check,
but their intersection is empty.  This closes scheduler-weight and tie-break
tuning; it is explicitly a finite-family result rather than an impossibility
theorem.

### `exp/fairness-metric-audit`

Question: does the frozen `younger-bypass >= 64` diagnostic identify the same
queries as extreme absolute sojourn and per-demand slowdown?

Result: no.  Across 16 domain-by-arrival-by-method cells, bypass64 is never a
strong detector for both user-facing tails.  Absolute sojourn and slowdown also
select different queries.  Future constrained schedulers must therefore report
both tails and treat bypass only as an ordering diagnostic.

### `exp/hard-fair-oracle`

Question: can a fixed completion/deadline utility retain the relaxed oracle
headroom when each pending query has a hard younger-bypass budget?

Result: yes within the registered four-point family.  HR uniquely selects
`B=32`; after freezing, Finance burst and Poisson both reduce mean sojourn by
about 8% and charged work by 5%--6% versus bounded CaGR, keep elapsed quality
regret no worse than frontier, and incur zero budget violations.  Roughly 30%
of slots are forcibly changed by the constraint.  The selected rule is causal
under the replay state interface, but remains pending joint user-tail,
cross-domain, and real-cost validation.
