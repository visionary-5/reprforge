# Qrel-free compression-risk compiler contract

Status: frozen before accessing the five remaining public ViDoRe v3 domains.
The contract applies to the new compression-risk direction. Earlier residual-
witness experiments remain exploratory evidence and do not count as method
validation under this protocol.

## 1. Research question

The paper question is not whether a multimodal index can be decoupled, pooled,
cached, pruned, or merged. It is:

> Given a query workload and a ladder of multi-vector index representations,
> can a compiler choose the cheapest representation whose ranking loss will
> remain below a declared tolerance on future queries, without relevance
> judgments, and abstain to a safer state when it cannot certify compression?

This separates three objects that must not be conflated:

1. **representation operator**: how vectors are pooled, pruned, merged,
   projected, quantized, or retained;
2. **risk estimator**: a qrel-free prediction of the ranking distortion caused
   by an operator at a given rate;
3. **compiler policy**: the representation chosen for each real physical
   decision unit under storage, build, search, and update budgets.

Decoupling is an enabling design. A new metric suite is an evaluation
contribution. Neither is the method contribution by itself. The paper-level
method must improve the risk--cost frontier over fixed compression rates,
qrel-free heuristics, and modern homogeneous compression operators.

## 2. Information boundary

The compiler may use:

- corpus embeddings and metadata available at index build time;
- an unlabeled fit-query workload and its query embeddings;
- full-reference or higher-capacity scores on fit queries, provided their
  one-time materialization cost is reported;
- operator costs measured without qrels;
- deterministic fit/validation roles frozen before evaluation.

It may not use:

- qrels, answer correctness, or downstream labels as compiler features;
- evaluation-query scores when compiling a static index;
- a per-query full-index fallback unless that full representation is actually
  resident and its bytes and token work are charged;
- dataset identity as a hand-written routing rule;
- thresholds retuned after a sealed-domain result is opened.

Qrels are loaded only by the evaluation program after the index and risk
decisions have been written. Full-reference rankings are teachers, not ground
truth: a compressed representation is allowed to outperform them on qrels.

The validation implementation enforces this boundary as separate processes,
not only as a convention inside one evaluator:

1. `tools/generate_vidore_unlabeled_surface.py` accepts only query and corpus
   shards and writes the full bank/runtime with `qrels_loaded=false`;
2. `tools/certify_compression_ranking.py` accepts only full and candidate
   runtimes and writes one immutable decision per rate;
3. `tools/materialize_vidore_labels.py` refuses to open the qrel shards until
   it has validated the dataset, full-runtime hash, stage, and
   `qrels_loaded=false` attestation of every supplied certificate;
4. only then does `tools/analyze_compression_risk.py` compute relevance-based
   quality and compare the already-written decision with the safety gate.

The post-certificate label manifest stores the SHA-256 of every prerequisite
certificate. This establishes an auditable ordering; merely placing labels in
a different array inside the same process is not sufficient.

## 3. Frozen data roles

The new method starts after the Computer Science residual result was observed,
so that collection cannot remain a sealed test for the new hypothesis.

| Role | Public ViDoRe v3 collection | Current access state | Permitted use |
|---|---|---|---|
| development | HR | opened | features, models, ablations |
| development | FinanceEN | opened | features, models, ablations |
| development | Computer Science | opened | features, models, ablations |
| validation | Industrial | qrels opened only after three certificates were hashed | one model-selection stage |
| validation | Pharmaceuticals | qrels opened only after three certificates were hashed | one model-selection stage |
| sealed transfer | Energy FR | unopened at freeze | final evaluation only |
| sealed transfer | Physics FR | unopened at freeze | final evaluation only |
| sealed transfer | Finance FR | unopened at freeze | final evaluation only |

The private Nuclear and Telecom collections are not assumed available. If they
become available, their role is additional sealed transfer. Collection-level
macro averages are primary; pooling all queries into one micro average is not.

Within each development or validation collection, workload queries use a
stable query-ID hash split. All stochastic procedures publish seeds. A source-
document or time split is preferred when the benchmark exposes a valid group;
random query cross-fit is described only as workload recurrence.

## 4. Representation ladder and comparisons

Every operator is evaluated as a measured rate--distortion curve, not at one
favorable point. The initial ladder contains:

- full ColPali late-interaction vectors as the high-capacity reference;
- semantic hierarchical token pooling from `colpali-engine`, already
  implemented in ReprForge, plus a separate fixed 2-D spatial control;
- canonical Light-ColPali / Light-ColQwen2 settings;
- Adaptive Grid Compression (AGC / OmniColPress) settings;
- Prune-then-Merge settings;
- Voronoi-style token pruning and the lossless/dominance baseline when code and
  model compatibility permit faithful reproduction.

Canonical paper settings are reproduced first. Cross-operator comparisons use
the closest measured points to common persistent-vector-byte fractions
`{1/32, 1/16, 1/8, 1/4, 1/2}`; no method receives a hidden denser point. Both
logical vector count and physical bytes are reported because projection and
quantization can change bytes per vector.

Required baselines are full, the cheapest corpus-wide cover, every reproduced
homogeneous operator/rate, a random or frequency-matched heterogeneous
allocation, and an oracle allocation marked as non-deployable. The risk
compiler is compared with the best fixed operator selected on validation, not
only with semantic hierarchical pool-9.

## 5. Decision units and valid coverage

Risk--coverage is meaningful only over units the compiler can independently
materialize. We recognize three levels:

- **configuration level**: one operator/rate for an entire collection;
- **physical-unit level**: independently stored document, shard, or declared
  cohort whose hybrid index is actually built and scored;
- **query level**: valid only for an explicitly dynamic serving policy with a
  resident fallback whose cost is included.

The main method targets physical units. Configuration-level transfer is a
required sanity check. Query-level selective curves are diagnostic unless the
system implements and charges the dynamic policy. Evaluation artifacts must
record `decision_unit` and `deployable_as_measured`; results with different
decision units are never placed in the same headline table.

## 6. Metrics

### 6.1 Retrieval effectiveness

For every collection and operating point:

- nDCG@5 and nDCG@10;
- Recall@100;
- paired candidate-minus-full and candidate-minus-cheap-cover differences with
  4,000-sample query bootstrap 95% intervals;
- collection macro average and worst-collection mean;
- the complete per-query values in a machine-readable artifact.

Full-relative regret for metric `M` is `M_full(q) - M_candidate(q)`. Positive
values are harm. Report mean, median, p90, p95, worst-5% CVaR, fraction harmed,
and catastrophic-harm rate. The frozen catastrophic thresholds are `0.10` for
per-query nDCG@10 regret and `0.05` for Recall@100 regret.

When full is better than the cheap cover, aggregate gap recovery is

```text
(M_candidate - M_cover) / (M_full - M_cover).
```

The denominator and its sign must be printed. Gap recovery is omitted when the
absolute denominator is below `0.005`; it is never averaged per query.

### 6.2 Qrel-free ranking fidelity

The evaluator compares candidate scores with the full-reference ranking using:

- Top-10 and Top-100 set overlap;
- retention of full Top-10 items in candidate Top-100;
- Rank-Biased Overlap at depth 100 with `p=0.95`;
- the rate at which a full Top-10 item falls outside candidate Top-100.

These metrics are available without qrels and can therefore be compiler
features on fit queries. They are not substitutes for final relevance metrics.
All ranking operations use stable corpus-position tie breaking.

### 6.3 Compression safety and calibration

The first configuration-level qrel-free ranking assurance gate was frozen on
2026-08-04 after inspecting the three development collections and before
opening validation data. It accepts an operator/rate only when all hold:

```text
mean full/candidate Top-10 overlap >= 0.90
one-sided 95% lower mean Top-10 overlap >= 0.88
one-sided 95% lower full-Top-10 retention in candidate-Top-100 >= 0.995
```

This is explicitly a development candidate, not a distribution-free relevance
guarantee. Its thresholds may not change on Industrial, Pharmaceuticals, or
the sealed French collections. A failure to transfer rejects the gate rather
than triggering threshold tuning.

### 6.3.1 Frozen physical compiler candidate

The first physical-method candidate was frozen after development-only
fit/evaluation on HR, FinanceEN, and Computer Science and before applying this
method to either validation collection. It is fixed as follows:

```text
cheap cover: semantic hierarchical pool-4 for every document
anchor state: full ColPali vectors retained in addition to the cheap view
anchor budget: 0.65 * full document-vector bytes
allocation: deterministic round-robin of
  (a) Top-10 membership-flip recurrence / anchor bytes and
  (b) competitive Top-100 rank displacement / anchor bytes
scoring: query-local affine full-minus-cheap residual completion
ridge: 1e-3
extrapolation: clip to the observed per-query anchor residual range
outer evaluation split: stable query-ID hash, 1/3 reserved
internal certificate: three-fold stable-hash cross-fit on outer-fit queries
```

For each internal fold, the anchor plan is compiled on the other two folds and
scored on the held-out fold without qrels. The three held-out candidate
surfaces are concatenated in original fit-query order and passed through the
unchanged qrel-free ranking assurance gate above. If that aggregate
certificate fails, the compiler emits a full-only abstention. If it passes,
the final anchor plan is refit on all outer-fit queries. Outer evaluation-query
scores do not affect the plan or abstention decision.

The fold salt is `physical-compression-crossfit-v1`, the compiler seed is
`20260804`, and certificate bootstrap uses 4,000 samples. No physical-method
threshold, budget, feature, or scoring parameter may change on Industrial,
Pharmaceuticals, Energy FR, Physics FR, or Finance FR. Validation qrels had
already been opened for the earlier configuration-level experiment, so the
physical-method validation is not called sealed; its compiler, bank, runtime,
and certificate must nevertheless be written by qrel-free processes before
physical-method relevance evaluation.

The frozen collection-level safety gate for an operating point is:

```text
one-sided bootstrap upper mean nDCG@10 regret <= 0.010
AND
one-sided bootstrap upper mean Recall@100 regret <= 0.010.
```

At a physical-unit coverage `c`, the deployed hybrid index compresses the
lowest predicted-risk fraction `c` and retains the safer state for the rest.
Its risk is measured by the resulting end-to-end ranking, not by summing local
page losses. Report:

- risk--coverage curve and area under it;
- excess area versus an oracle ordering and a random ordering;
- maximum safe compression coverage under the frozen gate;
- false-safe rate at each declared operating point;
- predicted-risk calibration by equal-count bins, including RMSE and maximum
  absolute calibration error.

Because ranking loss is non-separable, a unit may only be labeled safe after
constructing the corresponding hybrid index. Independent page `delta-nDCG`
labels are not accepted as the primary target.

### 6.4 Systems cost

Quality is paired with:

- persistent vector bytes and total on-disk index bytes;
- indexed vector count and MaxSim document-token work;
- end-to-end build time and extra compiler/teacher time;
- warm and cold search latency (median, p95, throughput) at fixed batch sizes;
- peak GPU and host memory;
- update amplification: bytes and time rewritten per changed source page;
- rollback metadata and storage overhead for a versioned index.

All ratios state their denominator. Logical savings inferred from token count
cannot replace measured physical bytes. Build/search timing is repeated after
warm-up and accompanied by the GPU, software versions, and concurrency.

## 7. Success and failure gates

The direction advances to a paper method only if one frozen compiler:

1. passes the safety gate on both validation collections without qrels;
2. after that selection is frozen, passes on at least two of three sealed
   French collections and does not catastrophically fail the third;
3. obtains strictly better safe compression coverage or byte--quality Pareto
   area than the best fixed modern operator;
4. reduces measured persistent bytes and MaxSim token work, not only a proxy;
5. retains its advantage under removal of any one proposed risk feature.

A result is a useful negative finding, but not the claimed method, if it only
works on HR/FinanceEN, requires per-domain qrel tuning, predicts average score
error without ranking safety, or shifts work from resident bytes into an
uncharged full-index fallback.

## 8. Artifact contract

Each run publishes:

- dataset and source hashes, model/adapter revisions, operator and rate;
- query/corpus IDs and score-surface hashes;
- qrel-free feature and decision files written before label evaluation;
- decision unit, split role, seed, and deployability flag;
- per-query quality and ranking-fidelity arrays;
- bootstrap seeds and all aggregate metrics;
- physical-cost measurements and raw timing samples.

Large model and dataset artifacts must pass an integrity gate before scoring:
expected immutable revision, exact byte size, upstream SHA-256, and a format-
level open/read check where supported. A repaired artifact is written to a new
path; a damaged shared path is not silently overwritten. Artifact repair is
experimental hygiene and is not counted as a method contribution.

Sealed results are appended once. Any later bug fix increments the artifact
schema, reruns every method affected by the bug, and preserves the original
result with an explicit invalidation note.
