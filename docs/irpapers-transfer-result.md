# IRPAPERS Transfer Result

## Decision

The independent transfer is **promising for progressive index construction**, but
it does not yet validate dynamic maintenance or a paper-level algorithm.

On the public [IRPAPERS benchmark](https://github.com/weaviate/IRPAPERS), a
real K=10 ReprForge execution visually materializes 511 of 3,230 pages, reaches
48.9/82.2/92.8 Recall@1/5/20, and completes construction plus retrieval in
118.1 seconds.  The controlled full-visual ColPali-v1.1 path takes 695.9 seconds
and stores 1.705 GB of visual vectors.  K=10 is therefore 5.89x faster and uses
6.16x fewer combined index bytes while improving Recall@1 by 10.0 points over
that full-visual path.

The more important comparison is the static full text+visual fusion using the
same scores.  K=10 improves Recall@1 by 0.56 point and gives up only 1.11 points
at Recall@5 and Recall@20 while visually representing 15.8% of the corpus.
This is a useful static Pareto point, not yet evidence that the current fixed-K
rule is the final contribution.

## Benchmark and accounting

[IRPAPERS](https://arxiv.org/abs/2602.17687) contains 166 information-retrieval
papers, 3,230 pages and 180 single-gold-page questions.  Every page supplies a
base64 page image and a GPT-4.1 transcription.  The supplied transcription is
treated as dataset input; ReprForge does not claim to avoid or accelerate its
original upstream generation cost.

The run uses:

- one NVIDIA A100-SXM4-80GB;
- PyTorch 2.5.1 with CUDA 12.4;
- the public
  [ColPali v1.1 checkpoint](https://huggingface.co/vidore/colpali-v1.1/tree/a0f15e3bcf97110e7ac1bb4be4bcd30eeb31992a);
- image batches of four, MaxSim batches of 16 and request batches of eight;
- the official IRPAPERS dataset revision
  `7d8ca2f6dd9efded3e27013d15782d584f93e9da`.

Model loading (8.2 seconds) and CSV validation/base64 decoding (29.4 seconds)
are recorded but excluded from every controlled construction-plus-retrieval
comparison.  Page image conversion inside each visual encoder call is charged.
The K=10 and K=20 completion times include BM25 index construction, visual page
construction, query encoding, candidate scoring and fusion.

The public query CSV contains one metadata contradiction at row 144:
`dataset_id=16_7`, while `pdf_id=15` and `page_number=7`.  The evaluator retains
the official `dataset_id` as the qrel so it follows the released leaderboard
contract, and records the contradiction.  One query corresponds to 0.56
percentage point, so future benchmark revisions should be checked.

## Controlled result

| Policy | Recall@1 | Recall@5 | Recall@20 | Visual pages | Index state | Build + retrieval |
|---|---:|---:|---:|---:|---:|---:|
| Local BM25 | 47.2% | 78.3% | 92.8% | 0 | 7.2 MB | 3.35 s build+score |
| Full visual ColPali v1.1 | 38.9% | 67.2% | 83.9% | 3,230 | 1.705 GB | 695.87 s |
| Static full z-score fusion | 48.3% | 83.3% | 93.9% | 3,230 | 1.712 GB | full-prebuild path |
| **Resident compiler K=10** | **48.9%** | 82.2% | 92.8% | **511 (15.8%)** | **277.0 MB combined** | **118.14 s** |
| Resident compiler K=20 | **48.9%** | 82.8% | 92.8% | 987 (30.6%) | 528.2 MB combined | 223.02 s |
| K=50 score replay | 47.8% | **83.9%** | **93.9%** | 1,799 (55.7%) | not executed | no deployable timing |

K=20 doubles the visual resident set and adds 105 seconds without improving
Recall@1 or Recall@20 over K=10.  K=50 recovers the static hybrid's deeper
recall but reduces Recall@1.  The measured failure is therefore not simply
"K is too small."  Different queries benefit from different actions, and
applying more visual intervention uniformly can harm the top rank.

The K=10 run contains 1,800 candidate events but only 511 unique page encodes.
Within-batch union removes 48.6% of events and cross-batch persistence produces
a 54.3% cache-hit fraction.  Its synchronous request-batch completion P95 is
10.6 seconds.  The 118.1-second aggregate is a cold-stream completion measure,
not an independently arriving query-latency claim.

## Rough position against published results

The [current official leaderboard](https://github.com/weaviate/IRPAPERS#retrieval-leaderboard-)
reports:

| Published system | Recall@1 | Recall@5 | Recall@20 |
|---|---:|---:|---:|
| BM25 | 45% | 71% | 90% |
| ColPali | 45% | 79% | 93% |
| ColQwen2 | 49% | 81% | 94% |
| Open static hybrid | 49% | 81% | 95% |
| Current closed hybrid | 58% | 91% | 98% |
| ReprForge K=10, this run | 48.9% | 82.2% | 92.8% |

This table supplies a coordinate, not a controlled leaderboard claim.  The
local BM25 is already stronger than the reported BM25, while the local
ColPali-v1.1 path is substantially weaker than the reported visual row.  The
official systems use different tokenization, models, fusion and serving code,
and do not publish comparable A100 construction times.  ReprForge is therefore
currently near the open hybrid's Top-1/Top-5 quality but behind its deep recall;
it has not beaten visual or hybrid SOTA.

## The algorithmic headroom

The frozen score surface supports a qrel-only diagnostic.  For each query, an
oracle chooses the cheapest successful action among no visual work and
candidate K in {10, 20, 50}.  This oracle is not deployable and is not reported
as a baseline result.

| Target | Oracle recall | Visual page-events | Action counts |
|---|---:|---:|---|
| Recall@1 | 52.8% | 110 | 85 no-build, 9 K=10, 1 K=20, 85 unresolved |
| Recall@5 | 86.1% | 170 | 141 no-build, 11 K=10, 3 K=20, 25 unresolved |
| Recall@20 | 94.4% | 150 | 167 no-build, 3 K=50, 10 unresolved |

Fixed K=10 spends 1,800 page-events.  The oracle reaches a higher Recall@1
with 110 events because most queries either need no visual intervention or
cannot be fixed by the available visual action.  This turns the next research
question from an abstract cache problem into a concrete action-value problem:

> Given a text retrieval state, should the index compiler return it unchanged,
> visually refine a small cohort, expand the cohort, or stop because the
> available representation is unlikely to change the useful ranking?

## Proposed next system mechanism

The next design should be an **estimate--verify--expand compiler**, not another
fixed K and not an eviction policy introduced before workload evidence exists.

1. **Estimate:** predict the value of actions {no-build, K=10, K=20, K=50}
   from runtime-visible text score margins, entropy, term coverage, query form
   and cohort overlap.  Train and evaluate by disjoint source papers so the
   system cannot memorize questions.
2. **Verify:** when the estimate is uncertain, build only a small visual probe
   and measure the actual score intervention.  Stop when the top ranking is
   stable or when the probe indicates that this visual representation is not
   useful.
3. **Expand:** pay for a deeper cohort only when the verified intervention can
   plausibly alter the requested cutoff.
4. **Commit:** persistence is a separate decision based on expected reuse and
   storage budget.  IRPAPERS cannot validate this stage because it has no
   document updates or natural temporal query trace.

This is the first algorithmic direction supported by a measured gap: the goal
is to approach the 52.8/86.1/94.4 oracle envelope without qrels and without
paying fixed-K work for every query.

## What remains before a paper claim

1. Replace or complement ColPali v1.1 with a current open visual retriever and
   add a strong dense-text+BM25 hybrid under the same cost accounting.
2. Pre-register a paper-disjoint train/validation/test split, then compare the
   estimator against fixed K, simple BM25-margin rules and the qrel-only oracle.
3. Run three query permutations.  Ranking semantics and final unique coverage
   are order-invariant here, but cold completion, batching and numerical noise
   are not.
4. Transfer the learned policy back to at least two ViDoRe domains and one
   harder text-overlap setting.  One 180-query benchmark is insufficient.
5. Measure downstream answer/evidence quality.  Recall alone does not prove
   that an agent can use the retrieved page.
6. Treat updates, deletion, versioning and admission/eviction as a later node.
   A temporal maintenance claim requires a public update/query trace; this run
   supplies none.

The current conclusion is therefore constructive but bounded: ReprForge has a
real independent Pareto result and a quantified selector headroom.  The next
contribution must be the deployable estimate--verify--expand mechanism and its
cross-benchmark evidence, not the observation that fixed K=10 happens to work
on IRPAPERS.

## Reproduction entry points

Paths are explicit so the repository does not assume a server layout or alter
an existing environment:

```bash
PYTHONPATH=. python -m tools.run_irpapers_transfer \
  --docs /path/to/irpapers-docs.csv \
  --queries /path/to/irpapers-queries.csv \
  --base-model /path/to/colpaligemma-base \
  --adapter /path/to/colpali-v1.1 \
  --output /work/results/summary.json \
  --actual-candidate-k 20 --replay-k 10 20 50

PYTHONPATH=. python -m tools.run_irpapers_resident \
  --docs /path/to/irpapers-docs.csv \
  --queries /path/to/irpapers-queries.csv \
  --base-model /path/to/colpaligemma-base \
  --adapter /path/to/colpali-v1.1 \
  --output /work/results/resident-k10.json --candidate-k 10
```

## Artifacts

- `reprforge/irpapers_benchmark.py`: official CSV validation, Recall evaluator,
  standard ColPali input profile, fusion replay and minimum-action oracle;
- `tools/run_irpapers_transfer.py`: full controlled matrix with incremental
  score-surface checkpointing;
- `tools/run_irpapers_resident.py`: actual resident-only Pareto point runner;
- `tools/analyze_irpapers_transfer.py`: raw-to-compact analysis;
- `results/systems/irpapers-transfer.json`: compact committed result.

Large page images, model files and the 4 MB score surface remain in the isolated
server workspace.  The raw summary, K=10 run and score surface SHA-256 digests
are recorded in the compact result.
