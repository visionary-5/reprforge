# Progressive visual materialization experiment matrix

Status: implementation protocol v0, 2026-08-07.  
GPU runs start only after data manifests, CPU dry-runs, and output-resume checks
pass.

## 1. Experimental questions

The matrix answers five questions in order.

1. **Capability:** does a visual retrieval representation discover evidence
   that a strong cheap locator cannot?
2. **Sparsity:** how much of Full visual retrieval gain can a physically built
   partial index recover at each materialization budget?
3. **Predictability:** can future-useful units be selected without test qrels
   or Full score surfaces?
4. **Amortization:** when does persistent materialization beat repeated
   query-time visual inspection, and when does it lose to Full?
5. **System effect:** do fewer builds/checks/bytes reduce end-to-end and tail
   latency on real hardware?

Failure at an early question changes the next method; it does not license
skipping later accounting or declaring the entire research problem dead.

## 2. Dataset roles

| Role | Dataset/domain | Purpose | Tuning rule |
|---|---|---|---|
| development | ViDoRe v3 Computer Science, Pharmaceuticals, Industrial | Existing score banks, physical implementation, feature and policy development | May tune only on explicit train/history folds. |
| sealed transfer | ViDoRe v3 Energy, Physics, Finance-FR | Cross-domain and multilingual test after v0 is frozen | No threshold, feature, budget, or fusion changes after labels/results are opened. |
| external regression | IRPAPERS | Text/image complementarity, scientific pages, strong public text/image baselines | Already used by older branches; cannot be called unseen. |
| granularity | MMDocIR layout-level | Page versus table/figure/text-region organization | Within-document retrieval limitation must be disclosed. |
| end-to-end | M3DocVQA/M3DocRAG first; MMDocRAG optional | Whether retrieval differences alter answer/citation quality | Run only after the retrieval protocol is stable. |

ViDoRe query order is dataset serialization, not a natural trace. Natural
order is retained only as one deterministic replay alongside controlled
synthetic orders.

## 3. Representation endpoints

All methods share the same corpus, parser outputs, query set, qrels, and final
verifier/reader when applicable.

| ID | Persistent state | Query-time visual work | Purpose |
|---|---|---|---|
| `T` | text/structure only | none | cheap lower endpoint |
| `DVI-k` | text/structure only | inspect top-k raw pages, discard outputs | never-materialize endpoint |
| `TransientCascade-k` | cheap visual/text locator | refine current candidates, discard high-fidelity state | LightSTAR-style nonpersistent endpoint |
| `Full-V` | Full visual retrieval for every page | final verification/reader only | eager endpoint |
| `StaticPartial-B` | Full visual retrieval for B% selected pages | optional raw-page fallback | static partial physical design |
| `Promote-B` | grows/evicts under B% capacity | inspect then optionally persist | progressive candidate method |
| `Oracle` | future-aware chosen pages | protocol dependent | undeployable headroom only |

Persisted retrieval embeddings do not replace final answer reasoning. The
reader cost common to all endpoints is reported but is not incorrectly
credited as avoided by materialization.

## 4. Baseline matrix

### 4.1 Cheap locators

- BM25 over supplied OCR/native text;
- dense text retrieval using one frozen public encoder;
- structure-aware BM25/metadata where the corpus exposes headings, drawing
  numbers, captions, or page hierarchy;
- best fixed fusion of the above selected on development only;
- compact visual locator when its complete physical build is charged.

`BM25` alone is not allowed to stand in for DVI's strongest cheap locator.

### 4.2 Static materialization

- SHA-256 random, repeated seeds;
- uniform document coverage;
- text scarcity/OCR confidence;
- visual complexity/type;
- cheap-locator disagreement;
- historical candidate frequency/page heat;
- discovery-risk coverage plus workload benefit;
- future-aware marginal-quality oracle.

### 4.3 Online admission and residency

- never promote (DVI endpoint);
- promote on first touch;
- promote on second touch;
- LFU/frequency admission with LRU, LFU, and GDSF eviction;
- residency-aware but quality-unconstrained greedy;
- ReprForge signed benefit/cost policy;
- Belady/future-aware admission oracle as an upper bound.

All policies use identical representation capacity and eviction machinery so
planner and cache benefits are not conflated.

### 4.4 Adjacent published systems

- DVI: implement the published structural/text locate-then-raw-page-analyze
  contract where dataset metadata permits; otherwise label the run DVI-like.
- EdgeRAG: reproduce or port its same-representation regenerate-and-cache
  mechanism only in the memory/regeneration experiment; do not imply it solves
  the text-versus-visual capability choice.
- AgenticOCR: use official code for downstream region parsing if its artifact
  and benchmark contract are reproducible; it is not a first-stage retrieval
  substitute.
- LightSTAR and MURE: run official artifacts when released and pinned. Until
  then report paper numbers separately and use clearly labeled mechanism
  baselines, never a claimed reproduction.

## 5. Physical budgets and workload axes

Static page budgets are `0, 1, 2, 5, 10, 20, 40, 60, 100%`. The 1/2% points
measure whether the oracle concentration is physically meaningful; 60% checks
whether a supposed partial method merely approaches Full.

All budgets are evaluated on frozen score surfaces. Direct builds are required
at 5/20/40/100%; the remaining points may interpolate cost only after direct
measurements establish the error of that interpolation. This avoids repeatedly
encoding the same pages merely to obtain a denser plot while still preventing
an unverified page-fraction projection from becoming the final cost claim.

Every dynamic policy is tested at 5/10/20/40% capacity under:

- deterministic dataset order;
- at least five random query permutations;
- Zipf page/document locality at exponents 0.8, 1.0, and 1.2;
- document/topic-clustered bursts;
- gradually broadening working set;
- abrupt distribution drift halfway through the trace;
- low-, medium-, and high-reuse horizons with repeated queries separated into
  train/history and future evaluation.

Synthetic replays are named synthetic. Their generator seed, page popularity,
query repetition, and drift point are saved in the result manifest.
Document-clustered replay and uniform document coverage are run only when the
export retains an authenticated source-document identifier; they are omitted,
not approximated from sequential page IDs, when that metadata is absent.

## 6. Metrics

### Quality and coverage

- nDCG@5/10, Recall@5/10/20/100, and query-hit at the verifier budget;
- candidate escape rate for each cheap locator and its visual repair rate;
- absolute Full-minus-method loss and worst 5% query loss;
- query-level quality violation rate at preregistered epsilon;
- visual gain recovery:

```text
(Q_partial - Q_text) / (Q_full_hybrid - Q_text)
```

Gain recovery is undefined/diagnostic when Full hybrid does not improve over
text; it is never replaced by a flattering ratio to Full.

### Physical cost

- parser/OCR, text index, compact visual index, and Full visual index build
  wall time and GPU-seconds separately;
- actual serialized bytes, vector count, resident bytes, and peak CPU/GPU
  memory;
- raw-page reads, decoded pixels, H2D bytes, VLM forward time, MaxSim/search
  time, fusion/reranking time, and end-to-end p50/p95/p99;
- cumulative cost versus query horizon, promotion stalls, evictions, rebuilds,
  and final materialized fraction;
- time to first usable index and time to reach 90/95/99% of final quality.

### Statistical reporting

- query bootstrap confidence intervals for quality deltas;
- mean and standard deviation over workload orders;
- document/domain macro results in addition to pooled queries;
- per-domain Pareto fronts and cross-domain worst case;
- all negative and timed-out runs retained in manifests.

## 7. Execution stages and gates

### Stage A: score-surface and data integrity, CPU

Validate hashes, qrel coverage, page IDs, ranking depth, selection leakage, and
metric parity. Generate every workload order and dry-run every baseline on a
tiny fixture. No direction-level decision is made here.

### Stage B: physical static curves, GPU

Build selected Full visual page banks at all budgets, without first building
and slicing a complete Full bank. Measure build and storage, then combine with
the fixed cheap locator and common verifier.

Gate: the oracle and at least one realizable policy must show a nontrivial
quality/cost gap over random and simple type/frequency selection in at least
two development domains. A selector failure retains the oracle bottleneck;
absence of oracle headroom closes partial materialization for that domain.

### Stage C: defer--promote workload crossover, GPU

Execute DVI-like transient inspection, static partial, simple promotion,
ReprForge, and Full on identical traces. Include complete raw-page I/O and
preprocessing in latency.

Gate: any progressive claim must beat DVI-like and Full in a nontrivial region,
beat fixed/static partial at the same quality/capacity, and avoid unacceptable
p95/p99 promotion spikes.

### Stage D: sealed transfer

Freeze features, normalization, admission, epsilon, budgets, and seeds. Run
Energy, Physics, and Finance-FR once. Failures trigger analysis, not test-set
retuning.

### Stage E: downstream answer use and scale

Run M3DocVQA/M3DocRAG answer quality and one larger corpus stress test. This
stage decides the scope of the final paper claim, not the existence of static
retrieval headroom.

## 8. Artifact contract

Each run writes:

```text
manifest.json       protocol/config/input/model/hardware hashes
events.jsonl        append-only query/build/promotion events
metrics.json        aggregate and per-order metrics
rankings.tsv        deterministic retrieval output
timings.jsonl       decomposed physical timing samples
stderr.log          failures and warnings
```

Large banks stay on attached storage and are copied or released only after
their manifest, rankings, metrics, byte counts, and hashes are synchronized to
the private repository or durable local storage. No wrapper overwrites an
existing run directory.
