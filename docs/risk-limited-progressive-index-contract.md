# Risk-Limited Progressive Multimodal Index

## Frozen contribution

ReprForge begins with a cheap corpus-wide text index.  A visual page
representation does not exist until the runtime explicitly acquires it.  For
every unbuilt candidate the runtime maintains a calibrated interval for the
score that full visual evidence could produce, constructs the pages whose
intervals still overlap the requested Top-k boundary, and stops when the
observed Top-k is separated from every unbuilt challenger.

The contribution is **decision-centric representation acquisition**, not an
oracle page classifier and not a cache replacement.  An oracle is reported
only as an undeployable headroom bound.  Standard GDSF manages already-built
representations in capacity-constrained experiments.

## Scope and prior-art boundary

The first implementation has two real construction states: corpus-wide BM25
and on-demand full ColPali.  Pooling a completed ColPali embedding can reduce
bytes and MaxSim work but cannot be called a visual-build saving.

Adjacent systems occupy important but different axes:

- [Col-Bandit](https://arxiv.org/abs/2602.02827) progressively reveals MaxSim
  cells after full document embeddings already exist.  ReprForge decides
  whether those document embeddings should be built at all.
- [MetaEmbed](https://arxiv.org/abs/2509.18095) and
  [MURE](https://arxiv.org/abs/2603.13349) provide multi-resolution
  representations.  A representation ladder alone is not claimed as novel.
- [LightSTAR](https://arxiv.org/abs/2606.23539) establishes lightweight
  candidate selection followed by expensive refinement.  Fixed candidate
  selection is therefore a required baseline, not the proposed mechanism.
- Binary resident/transient retention is covered by strong caching baselines
  and was rejected in ReprForge V4.  Retention is not part of this claim.

## Deployable score contract

For each query, BM25 selects a deterministic Top-100 pool.  The base score is
the BM25 z-score inside that pool.  Raw ColPali MaxSim is divided by the
query's lexical token count, then standardized with mean and variance fitted
only on training source groups.  Final score is the equal-weight sum of base
and standardized visual score.

No test qrel, test visual score, page identifier, dataset identifier or
content-type regular expression is a policy feature.  The cheap feature view
contains BM25 score/rank, query and OCR-text lengths, and query-level BM25
margin/dispersion.

If fully observed deployment scoring loses more than 0.01 Recall@5 or
nDCG@10 against the previous candidate-relative teacher, the preregistered
fallback is a query-conditional affine normalization fitted from query length
and BM25 distribution features.  Test thresholds or content-specific patches
are forbidden.

## Risk model and acquisition

Two fixed-penalty ridge models estimate normalized visual score and log
absolute residual.  The decision certificate calibrates a one-sided
split-conformal multiplier from the maximum positive standardized residual
across all Top-100 candidates of each calibration query.  This is the exact
event needed by stopping: no hidden challenger exceeds its upper bound.
A lower multiplier is retained only for diagnostics.  The default risk is
`alpha=0.05`.  The resulting statement is query-level marginal simultaneous
upper coverage under exchangeability; it is not conditional, temporal or
adversarial coverage.

The score-envelope path is followed by a decision-level alternative.  A
weighted ridge model predicts full-score Top-k membership from the same cheap
features; split-conformal calibration sets the inclusion score that covers
every Top-k member of each calibration query.  The runtime builds that
variable-size candidate set and scores every returned page exactly.  Because
candidate-set coverage is adjacent to certified pruning, it is compared
directly rather than silently relabelled as a novel score certificate.

The runtime first observes the predicted Top-k.  It then repeatedly acquires
unobserved candidates whose intervals overlap the current exact kth score,
ordered by interval width divided by measured build cost.  GPU work is issued
in batches of four.  It stops only after all returned pages have exact visual
scores and the kth score is no smaller than every unobserved upper bound.  If
the intervals never separate, it builds the entire pool and returns the exact
pool-local result.

## Evaluation contract

IRPAPERS is the development workload because it has a frozen full 180 x 3,230
score surface and 19 source-paper groups.  Five balanced source-paper folds
are used.  For test fold `f`, fold `(f+1) mod 5` calibrates conformal risk and
the remaining three folds fit the models.

Required baselines are BM25, full visual prebuild, fixed K in
`{5,10,20,50,100}`, the previous stability/margin policy, BM25-margin routing,
boundary admission, a certified-pruning-style threshold and an explicitly
undeployable teacher oracle.  All candidate policies share Top-100 candidate
membership.

The offline gate requires:

- at least 93% held-out query-level simultaneous upper coverage;
- at most 5% exact Top-k-set disagreement;
- Recall@5 within one IRPAPERS query (0.56 point) of the full-pool teacher;
- at least 20% fewer candidate events and unique builds than the cheapest
  fixed K that attains the same Recall@5.

The A100 gate additionally requires at least 20% less measured visual build
time and 1.15x lower end-to-end time in two runs.  The frozen algorithm then
transfers without threshold repair to official ViDoRe v3 HR English and
Finance-EN.  A paper-level continuation needs the offline quality/resource
gate on IRPAPERS and at least one ViDoRe dataset, plus 1.15x end-to-end speedup
on two datasets.

## Reproduction entry point

```bash
PYTHONPATH=. python -m tools.analyze_risk_limited_acquisition \
  --score-surface /path/to/irpapers-full-runtime-score-surface.npz \
  --documents /path/to/IRPAPERS-documents.csv \
  --queries /path/to/IRPAPERS-queries.csv \
  --output /tmp/risk-limited-irpapers.json
```

The compact output records every split, conformal multiplier, coverage,
teacher agreement, retrieval quality and construction event count.  Raw
images, embeddings and score matrices remain outside Git.
