# Research direction decision — 2026-08-03

## Update — 2026-08-04: anytime cohort compilation

Post-hoc physical compression and qrel-limited risk auditing did not produce a
safe cross-domain compiler. The surviving positive node moves the objective
back to the original build-time bottleneck: publish cheap evidence immediately
and schedule reusable visual cohort construction by the query--page dependency
frontier.

The first qrel-free scheduler passes an exact-work gate on HR and Finance-EN,
reducing mean encoded pages at query completion by 19.3% and 28.7% versus
FIFO. On a real HR A100 run it completes in 91.85 seconds versus 98.03 for
FIFO and 99.26 for static popularity, while cutting first publication from
15.54 seconds to 6.73 seconds. Finance frontier transfers at 172.85 seconds
versus 190.50 for FIFO; its static-popularity recovery is still pending.

This does not make asynchronous execution, caching, or query grouping novel.
The candidate paper node is the evidence-quality trajectory of a reusable
physical multi-vector index under charged construction work. It remains
provisional until bounded arrival windows, answer stabilization, and the
per-query revision-harm problem are evaluated. See
[`cohort-frontier-scheduler-result.md`](cohort-frontier-scheduler-result.md).

## Update — 2026-08-04: compression risk is now the leading test

The next hypothesis and its metrics are frozen in
`compression-risk-contract.md`. HR, FinanceEN, and Computer Science are
development domains for this new hypothesis; the latter was already opened to
falsify residual witnesses and therefore cannot remain sealed. Industrial and
Pharmaceuticals are validation; Energy FR, Physics FR, and Finance FR remain
sealed transfer.

The first three-domain pool-9 replay identifies a consistent phenomenon. At
11.068% of full vectors, pool-9 retains 99.873% of full Top-10 documents inside
its Top-100, but only 78.28% at Top-10. Its macro mean nDCG@10 regret is .0206,
while worst-5% query CVaR is .3525 and 17.8% of queries lose more than .10.
Pool-9 fails the conservative safety gate on all three domains.

The target is consequently not generic score reconstruction or another claim
that aggressive pooling is nearly lossless. It is qrel-free prediction of
competitive-boundary distortion, followed by physical capacity allocation or
abstention. The full result is in `compression-risk-baseline-result.md`.

The completed pool-25/9/4 curve makes the immediate target narrower. Pool-4
uses 24.95% of vectors and has mean nDCG@10 regret below .01 on all three
development domains. It passes the frozen safety gate on Computer Science but
misses on HR and FinanceEN with one-sided upper regrets .0137 and .0140. No
compressed fixed rate passes all three domains. Pool-4 is therefore the base
for the first risk-compiler test: selectively recover the small certification
gap without giving up its 75% byte reduction. Tail CVaR remains .168--.260, so
mean regret alone cannot define success.

Both held-out validation collections now support the frozen risk signal.
Before qrels were opened, the unchanged gate accepted Pharmaceuticals pool-4,
abstained on Pharmaceuticals pool-9/25, and abstained on all three Industrial
rates. Post-certificate evaluation made the same six decisions. The selected
plan uses 62.48% macro resident capacity versus 100% for the best fixed state
that is safe on both collections, with macro mean nDCG@10 regret -.00007.
Industrial pool-4 is an important counter-case: despite 100% retention of the
full Top-10 inside candidate Top-100, its nDCG@10 and Recall@100 upper regrets
are .01806 and .02119, so candidate coverage alone cannot certify safety.

Configuration-level transfer is therefore established, but it is not yet a
paper method. The immediate target is a physical document/shard compiler that
uses only a fit workload, materializes a real hybrid index, and improves the
safe capacity frontier on future queries. Industrial is the first target:
global pool-4 is unsafe, leaving room to test whether selective full-capacity
upgrades can recover safety without falling back to a fully resident index.

### Physical-compiler development update

A first physical candidate now passes the frozen relevance safety gate on
held-out query-hash splits of all three development domains at approximately
89.9% of full persistent document-vector bytes. It stores a pool-4 cover plus
a parameter-free boundary coreset of full-document anchors, then uses paired
anchor scores for query-local affine residual calibration. Mean nDCG@10 upper
regrets are .00738 (HR), .00619 (FinanceEN), and .00829 (Computer Science).
Raw heterogeneous score mixing fails all three at matched allocation, while
random anchors with the same calibration fail HR and FinanceEN.

The Computer Science dual-view bank has been physically materialized and
scored: 322,570,240 bytes versus 358,604,800 full, with materialized anchor
scores matching the prior full surface to maximum absolute error `1.91e-6`.
The full evidence and limitations are in
`physical-compression-compiler-result.md`. This is a promising method
candidate, not a paper claim: nested qrel-free abstention, physical validation,
sealed transfer, modern operators, and optimized systems measurements remain.

The later frozen physical validation rejects this candidate as the final
method. Industrial and Pharmaceuticals both pass the internal and reserved-
query qrel-free certificates, but their relevance nDCG@10 upper regrets are
.01086 and .01104, just above the .01 tolerance. Both actual banks have
89.93--89.94% of full document-vector bytes and anchor/full score parity within
`1.91e-6`, so the result is a genuine 2/2 certificate false-safe rather than
an implementation error. No validation threshold is retuned.

The subsequent residual-matrix branch is also rejected. The five opened
collections have entropy effective ranks 58.5--101.9 after row centering and
standardization; rank 64 explains only 65.7% of Industrial residual energy.
Ordinary pivoted-column ridge completion does not transfer, and truncated SVD
at rank 64 still misses the HR nDCG@10 safety tolerance with 128 and 256
anchors. More Frobenius-oriented matrix reconstruction is not the next use of
GPU time.

The leading problem is now **label-efficient risk-limiting physical index
compilation**. An unlabeled workload still generates a ladder of actual
heterogeneous banks and supplies ranking-distortion proxies. A deliberately
small, disjoint relevance sample audits that ladder and selects the cheapest
plan whose estimated relevance regret is safe; otherwise the compiler emits
the full-index fallback. This does not claim conformal or risk control itself
as novel. The systems contribution must be the joint physical-design decision
and its measured byte/build/update consequences.

The new primary metrics are (1) false-safe probability over repeated
calibration/audit splits, (2) label count required to match the full-label
oracle plan, (3) selected persistent bytes and build/update work, and (4)
mean plus worst-tail retrieval regret. A qrel-free gate and uniform random
label sampling are required baselines. The immediate method hypothesis is
qrel-free boundary-stratified auditing: oversample queries with observed
ranking distortion while retaining a probability sample from every stratum,
then use a design-aware estimate rather than treating the biased sample as the
workload mean.

### Risk-audit and homogeneous-token verdict

The first label-efficient audit implementation is not a surviving method.
Qrel-free distortion does enrich positive-harm queries by roughly 1.8--3.6x
on HR, but strict plan-fit/calibration/audit rotations are unstable. The
full-label audit oracle varies from full to a 75%-byte plan across rotations;
at 64 calibration labels the boundary-stratified selector's false-safe rate
ranges from 1.5% to 97.5%. More labels can increase false-safe decisions when
the calibration slice experiences compression gains that do not recur in the
audit slice. Ordinary mean-risk estimation is therefore not the paper method.

A second bounded probe removed the heterogeneous-score confound entirely. It
used workload query prototypes to select a homogeneous fixed-budget subset of
original document tokens, whose MaxSim score is pointwise no greater than the
full score. This also fails decisively on Computer Science: at 64/128 tokens
(6.21%/12.43% of full), nDCG@10 is .5439/.5994 versus .6285/.6688 for matched
random and .7027 for pool-9. Workload saliency alone destroys document-side
discriminative coverage. No merge extension is run because the frozen
matched-random gate fails.

These failures close two tempting shortcuts: a statistical gate cannot rescue
an unstable physical plan, and empirical query saliency cannot replace a
document-coverage compression operator. The next decision should move back to
the user's original build-time motivation: post-hoc compression consumes a
full encoded bank and therefore cannot reduce time-to-first-usable RAG index.

## Decision checkpoint

There is not yet a defensible paper method. The earlier
**workload-conditioned residual witness index** is now a falsified prototype,
not the leading claim: it is strong on HR, partially transfers to Finance, and
fails a sealed Computer Science transfer. Probe quantization, adaptive probe
counts and scalar residual sketches do not repair the third-domain failure.

The research direction remains heterogeneous multimodal index compilation,
but the method question is now sharper:

> Given a workload and a ladder of modern multi-vector compression operators,
> compile the cheapest representation only when a qrel-free risk signal says
> its ranking distortion will transfer; otherwise allocate more capacity or
> abstain to a safer representation.

The claim cannot be decoupling, routing, caching, BM25, RRF, multimodal fusion,
token pruning, token merging, or workload conditioning by itself. A paper must
contribute either a reliable compression-risk certificate/selector that works
across the public ViDoRe v3 domains, or a new representation operator that
beats the 2025--2026 compression baselines. Until one of those gates passes,
ReprForge is an evidence-rich research substrate rather than a paper result.

## Evidence obtained today

The Heterogeneity Atlas compares a best global route, label-free portfolios, a
fit-label static document plan, and a held-out query-route oracle. Mixed-route
scores use query-local rank calibration; all label-using results are marked as
diagnostic.

| Collection | Best global | Query oracle gap | Static-document gap | Best-rank portfolio |
|---|---:|---:|---:|---:|
| ViDoRe v3 HR, nDCG@10 | 0.4880 | +0.1038, CI [0.0730, 0.1388] | -0.0202, CI [-0.0522, 0.0086] | 0.5072 |
| ViDoRe v3 Finance, nDCG@10 | 0.4524 | +0.1142, CI [0.0735, 0.1589] | +0.0268, CI [0.0072, 0.0489] | 0.4779 |
| Biomedical interaction pilot, nDCG@5 | 0.7097 | +0.0307, CI [0.0086, 0.0569] | -0.0518, CI [-0.1096, -0.0034] | 0.7309 |

The route disagreement is large: held-out text/visual Top-10 Jaccard is 0.372
on HR and 0.230 on Finance. The best alternative route rescues about 11% of
relevant mass at Top-10 on both collections.

### Falsified shortcuts

1. **Static page type is sufficient.** False on HR and biomedical; only
   Finance shows a small collection-specific static gain.
2. **A shallow query router recovers the oracle.** False. Five-fold cross-fit
   lexical, benchmark metadata, and cheap-score statistics do not beat the
   best global route on HR or Finance.
3. **A few query-local scalar landmarks recover visual behavior.** False. At
   8/20 probes, the completion model recovers only 31% of full cohort gain on
   HR and has negative gain recovery on Finance. At least 12–16 probes are
   needed for useful recovery.
4. **Candidate identity always fixes routing.** False as a general claim. It
   improves Finance by +0.0256 nDCG@10 under random-query cross-fit, but loses
   on HR and biomedical. All 309 Finance queries form one connected component
   through shared relevant source documents, so a source-document-disjoint
   estimate is not identifiable. The positive result is evidence for workload
   specialization, not unseen-document generalization.

### Existing positive mechanism

The prior candidate-relative compiler remains the strongest method result:

- HR: 0.5373 nDCG@10 while materializing 895/1,110 visual pages;
- Finance: 0.5628 nDCG@10 while materializing 1,855/2,942 visual pages;
- the same frozen `K=20` and fusion equation are used on both collections.

Its limitation is also unchanged: admit-on-first-touch grows to 63–81% of the
corpus and cold queries wait for expensive construction. Basic caching is not
novel, and GDSF already closes the earlier two-state cache result.

## Relation to closest work

- [UniversalRAG](https://universalrag.github.io/) routes queries among
  modality- and granularity-specific corpora. It assumes those corpora have
  already been built; it does not optimize which document representations to
  materialize under a workload.
- [R3AG](https://arxiv.org/abs/2604.22849) learns query-specific retriever
  routing from document assessments and answer correctness. Its action is a
  retriever, not an index-time representation state.
- [RAGRouter](https://arxiv.org/abs/2505.23052) argues that retrieved documents
  alter routing decisions. This agrees with our query-only negative result,
  but its target is routing among RAG-augmented LMs rather than multimodal
  representation construction.
- The [multi-granularity multimodal retrieval
  framework](https://arxiv.org/abs/2505.01457) constructs page, region, OCR,
  and visual signals, then fuses and verifies candidates. It is a strong
  quality baseline, but it does not treat representation materialization as a
  workload-constrained physical-design problem.
- [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) builds text and visual
  indexes and applies GMM-based dynamic hybrid retrieval plus iterative
  reasoning. It is a required end-to-end baseline, not a representation
  compiler.

The reviewed closest work makes generic “multimodal routing” too crowded for
a contribution. The less occupied gap is the joint optimization of cohort
ranking semantics and physical representation materialization.

## AnythingLLM interpretation

The product fork already supplies a serious systems substrate:

- immutable, resumable and atomically published index generations;
- T (OCR/native text), V (direct visual), and S (semantic sketch) branches;
- hardware-specific cost measurement, replay, fusion, diagnostics and gates;
- a page-level LightGBM CARS scheduler.

The scientific weakness is the CARS abstraction: predicted gain is learned
from static page features and scheduled independently. Atlas and prior exact
interventions show that this is not generally sufficient. AnythingLLM should
therefore remain the later execution platform; its current page router should
not define the paper method.

## Proposed problem

For a cheap base index and query workload `Q`, let `C(q)` be the candidate
cohort and `x[d,r]` indicate whether representation `r` of document `d` is
materialized. The compiler chooses physical states under build, memory,
maintenance and cold-latency budgets:

```text
maximize    sum_q U_q(C(q), x[C(q), :])
            - build_cost(x) - memory_cost(x) - maintenance_cost(x)
subject to  cold-query and storage budgets
```

`U_q` is a listwise cohort utility, not a sum of page gains. Cohorts overlap,
so a physical representation constructed for one query can be reused by later
cohorts. This is a workload hypergraph physical-design problem with
non-separable ranking semantics.

## Cohort-certificate decision

The bounded algorithmic test was **not** another scalar utility predictor.
For each training query, use the complete representation surface only as a
teacher and find a small **cohort certificate**: a subset of expensive
candidate states sufficient to reproduce the teacher's Top-k result under a
frozen partial-fusion contract. Then learn or cache certificate patterns from
cheap cohort-observable features.

This changed the supervision target from independent `delta-nDCG(page)` to a
set-valued ranking certificate. It directly respects the observed
non-additivity. Teacher qrels are not required for certificate construction;
qrels remain evaluation-only.

The result is depth-dependent rather than a binary pass/fail:

| Collection and base | Exact target | Median exact states | Unique exact states / candidate union | Base → teacher |
|---|---:|---:|---:|---:|
| HR text, K=20 | ordered Top-5 | 7 | 762/932 (81.8%) | 0.4727 → 0.5059 nDCG@5 |
| Finance text, K=20 | ordered Top-5 | 8 | 1,068/1,619 (66.0%) | 0.3958 → 0.4456 nDCG@5 |
| IRPAPERS BM25, K=20 | ordered Top-5 | 7 | 517/987 (52.4%) | 0.7833 → 0.8222 Recall@5 |
| MMDocIR pool-25, K=10 | ordered Top-5 | 4.5 | 158/256 (61.7%) | 0.5962 → 0.5999 nDCG@5 |

Exact ordered Top-10 is too expensive on HR and Finance: medians rise to
12/20 and 13/20, and unique-state coverage rises to 90.7% and 83.8%. Exact
set preservation alone is misleading because it changes internal order and
retains only 44.5%/76.0% of the full nDCG@10 gain.

The most important candidate-generation control comes from MMDocIR. A text
Top-10 cohort has an nDCG@5 ceiling of only 0.0850. Replacing it with a global
25-token pooled-visual base raises the base to 0.5962 while using 16.4 MB,
versus 411.9 MB for the full visual route. Therefore selective refinement is
viable only after a cheap representation preserves candidate recall.

## Rejected selector variants

All selector probes use query-holdout evaluation and no qrels for fitting:

- A ridge selector over cheap ranks, gaps, margins and score-profile features
  does not reliably beat fixed Top-B selection.
- A query-local boundary policy based on three exact landmarks needs 12–16 of
  20 probes on HR/Finance to approach the teacher.
- A learned completion-plus-active-acquisition policy recovers only 62.1% and
  35.3% of held-out HR/Finance gain at B=8. It reaches the IRPAPERS teacher at
  B=8 but does not transfer as a general mechanism.
- A leave-one-state-out stability gate prevents the worst low-budget drops,
  but accepts too few HR/Finance/IRPAPERS queries at larger budgets and discards
  most of the available gain. Stability is useful as a feature, not a complete
  safety certificate.

These results reject a paper centered on a one-shot page classifier or a
generic low-budget active reranker. The missing signal is not recoverable from
cheap scalar scores alone.

## Surviving method: workload representation ladder

The stronger unit is the **workload**, not one query. Fit-query certificates
form a hypergraph over reusable document states. A physical compiler can build
a compact corpus-wide base, rank candidate full states using query-log
frequency or qrel-free teacher fidelity, and serve unseen queries with a small
fixed probe allowance.

Initial query-holdout results at the largest tested offline budget are:

| Collection | Selected fit plan | Offline full-state cost | Cost after eval anchors | Held-out gain recovery |
|---|---|---:|---:|---:|
| HR | certificate frequency | 40.0% | 49.1% | 95.2% |
| Finance | certificate frequency | 28.5% | 31.8% | 81.1% |
| IRPAPERS | candidate frequency | 24.9% | 25.8% | 100.0% |
| MMDocIR pool-25 | no refinement is favored | 0% | 0% | teacher gain is only 0.00485 |

The current plan-family selector is not the final algorithm. At low budgets,
fit teacher fidelity can overfit and HR/Finance quality can fall below the
base. The paper method now needs a conservative optimizer with a no-upgrade
action, a base-quality safety constraint, and a genuinely listwise marginal
objective. This is the next method work, not another routing model.

## Updated execution order

1. Replace heuristic plan-family selection with a conservative listwise
   workload optimizer and validate it on the frozen four-dataset surface.
2. Add source-/time-grouped workload splits where identifiable; report random
   query holdout only as workload recurrence, not unseen-document transfer.
3. Freeze a representation ladder baseline: text/BM25, pooled visual, and full
   visual, with candidate recall and final ranking reported separately.
4. Run real versioned-index cold-start, time-to-quality, storage, update and
   rollback experiments in the existing MMDocIR/AnythingLLM substrate.
5. Add answer-level RAG evaluation only after the retrieval/cost Pareto frontier
   is stable; do not use generation variance to rescue weak retrieval claims.

## Current paper thesis

> Multimodal RAG indexes should be compiled as a workload-conditioned
> representation ladder: a cheap corpus-wide state preserves candidate recall,
> while reusable exact deltas are materialized only where overlapping
> competitive cohorts justify their physical cost.

This thesis is evidence-backed but not paper-complete. Small exact Top-5
certificates exist, but they are not predictable from cheap scalar features.
The decisive remaining question is whether a conservative workload optimizer
can turn their overlap into a stable held-out quality/cost frontier.

## Later update: capacity is a regularizer, not only an approximation

The recovered five-state MMDocIR route bank changes the method target. The
states are `text`, hierarchical visual pools 25/9/4, and full visual. On the
30-document, 4,643-item, 141-query within-document surface, full visual is not
the best uniform state:

| Uniform state | Storage / full | all-query nDCG@5 |
|---|---:|---:|
| text | 5.51% | 0.1035 |
| pool-25 | 3.98% | 0.5257 |
| pool-9 | 11.07% | **0.5601** |
| pool-4 | 24.95% | 0.5531 |
| full visual | 100% | 0.5514 |

The non-monotonicity is not noise that a full-teacher fidelity objective should
erase. Lower-capacity routes can suppress MaxSim distractors and improve the
ranking. A qrel-free compiler that bounded every state's normalized score
error against full visual was therefore rejected: at 11.8% storage it reached
only 0.4968 held-out nDCG@5, below uniform pool-9 and the type policy.

An exact cost-constrained listwise search over all 625 mappings from the four
observable content types (`formula`, `image`, `table`, `text`) to the five
states is the first positive compiler result. At a nominal 6% budget:

- query-recurrence five-fold cross-fit: 0.5948 nDCG@5 at 5.41% mean storage;
- source-document-disjoint five-fold cross-fit: 0.5922 at 5.29% mean storage;
- versus uniform full on the source-document protocol: +0.0408, paired 95% CI
  [0.0117, 0.0726];
- versus uniform pool-9: +0.0321, paired 95% CI [0.0047, 0.0615].

The selected mappings are stable in their important decisions: text and
formula use pool-25, images use pool-9, and tables use pool-4 or pool-9. The
previous frozen Typed-Capacity rule (full tables, pool-9 images, pool-25 other)
scores 0.5964 on all 141 queries at 8.81% storage, but the new result learns the
mapping inside every training fold and generalizes to held-out source
documents at lower cost.

A more flexible ten-group policy split images and tables by grayscale entropy
and edge energy. Cost-regularized coordinate search reached 0.5890 at 6.50%
storage under source-document cross-fit, which is dominated by the simpler
four-type search. This rejects static low-level visual complexity as the next
source of method complexity.

### Consequence for the paper

The paper should no longer present full visual as an unquestioned teacher. The
revised central mechanism is **capacity allocation as listwise index
regularization**:

> A multimodal physical-design compiler should choose each representation
> state for its effect on evidence--distractor competition, not for pointwise
> similarity to the largest representation.

This makes cohort certificates a safety/diagnostic tool rather than the main
optimization target. It also gives a sharper novelty boundary against nearby
work:

- [MetaEmbed](https://arxiv.org/abs/2509.18095) exposes a uniform test-time
  token-count control through a specially trained representation; the compiler
  here assigns different existing states to different indexed objects under a
  workload and physical budget.
- [LightSTAR](https://arxiv.org/abs/2606.23539) performs lightweight candidate
  selection followed by adaptive refinement; it does not compile a persistent
  heterogeneous physical index.
- [Reminisce](https://pubmed.ncbi.nlm.nih.gov/40537518/) uses query-driven
  coarse-to-fine embeddings on device; its main target is online embedding and
  hardware efficiency rather than listwise per-object physical design.
- [Col-Bandit](https://arxiv.org/abs/2602.02827) prunes MaxSim cells at query
  time without changing the index; it is complementary to offline state
  assignment.

### Remaining validity boundary

The positive result is not yet a paper-complete claim:

1. MMDocIR uses official within-document candidate sets, only 30 documents and
   141 queries. A full-corpus ViDoRe transfer is mandatory.
2. Hierarchical pooling reuses the full visual encoder, so the current win is
   in index bytes and likely scoring work, not visual construction time. A
   build-aware ladder needs at least one state that genuinely avoids or reduces
   visual encoding.
3. The listwise compiler uses fit qrels, interpretable operationally as judged
   historical queries or clicks. A weak-supervision or feedback-free variant
   is desirable, but should not replace the honest supervised result with a
   weak full-teacher surrogate.
4. Exhaustive type search is a strong algorithmic baseline, not sufficient
   novelty by itself. The final method needs scalable state assignment while
   preserving the source-document-disjoint gain and the no-upgrade option.

## Decisive update: pooled cover plus residual token witnesses

Full-corpus ViDoRe measurements reject page-level route replacement but expose
a stronger token-level construction. Hierarchical pool-9 uses 11.07% of full
tokens on both HR and Finance while preserving a high-recall substrate:

| Dataset | Recall@100 pool-9 / full | nDCG@10 pool-9 / full |
|---|---:|---:|
| HR | 0.8801 / 0.8784 | 0.4931 / 0.5178 |
| Finance-EN | 0.8229 / 0.8273 | 0.4503 / 0.4732 |

The current post-hoc pooling implementation is not a construction-speed
method: HR pool-9 takes 305.2s versus 117.7s for full, and Finance takes
805.5s versus 281.6s. It is a storage/scoring substrate.

The new compiler starts from the pooled tokens. For every fit-workload query
token and page, it records the full token attaining MaxSim only when that token
beats the pooled cover by more than residual threshold `epsilon`. The union of
those witnesses is appended to the pooled page representation. Qrels are never
used. Five deterministic query-hash outer folds compile on past queries and
evaluate the physical token set on unseen queries.

| HR configuration | tokens / full | nDCG@5 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|
| pool-9 only | 11.07% | 0.4703 | 0.4931 | 0.8801 |
| epsilon 0.30 | 11.25% | 0.4797 | 0.5044 | **0.8845** |
| epsilon 0.20 | 12.01% | 0.4809 | 0.5120 | 0.8830 |
| epsilon 0.15 | 13.80% | 0.4939 | 0.5155 | 0.8700 |
| epsilon 0.12 | 16.99% | **0.4951** | **0.5211** | 0.8705 |
| full | 100% | 0.4942 | 0.5178 | 0.8784 |

At 13.80% and 16.99% tokens, nDCG@5 and nDCG@10 are statistically
indistinguishable from full and significantly above pool-9. At 11.25%, the
nDCG@10 gain over pool-9 is +0.0112 with paired 95% CI [0.0009, 0.0212], and
the Recall@100 gain is +0.0044 with CI [0.0003, 0.0086]. Matched random full
tokens are worse at every measured capacity.

Two rejected controls sharpen the mechanism. Workload-wide raw MaxSim winners
need 50--65% tokens to match full, while a Top-K-only witness set destroys
unseen-query recall. The pooled cover is essential; sparse residual witnesses
repair detail without replacing the coarse retrieval substrate.

Token pruning itself is prior art. SIGIR 2025 studies dominance/lossless
ColBERT pruning; Light-ColPali (Findings ACL 2025) establishes token merging as
the strong VDR compression baseline; SIGIR 2026 estimates token influence with
uniform-sphere Voronoi samples. The prospective novelty is narrower: empirical
workload rather than uniform-query geometry, a pooled cover plus sparse exact
MaxSim residuals, heterogeneous per-page cardinality, and evaluation as a
physical multimodal index under held-out workload.

The August 3 literature refresh raises the baseline bar further. [Multi-Vector
Index Compression in Any Modality](https://arxiv.org/abs/2602.21202) introduces
OmniColPress attention-guided clustering with learned universal query tokens
and evaluates text, visual-document and video retrieval. [Prune-then-Merge
(Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1247/) reports
consistent gains across 29 VDR datasets. Any residual/compiler paper must
compare against both; pool-9 alone is no longer an adequate state of the art.

The HR epsilon curve was inspected during development. The registered next
gate is frozen transfer of `epsilon in {0.12, 0.15, 0.20, 0.30}` to Finance-EN,
then Light-ColPali/Voronoi baselines and measured compact-index build/search.

## Three-domain residual verdict

The Finance frozen gate failed. Its best raw residual point was
`epsilon=0.15` at 13.19% tokens: nDCG@10 0.4595 versus 0.4503 pool-9 and
0.4732 full. The gain over pool-9 was not significant and the loss to full was
significant. A hard full-Top-K boundary mask looked strong on HR but fell below
pool-9 on Finance, demonstrating collection-specific workload overfitting.

Spherical query-token probes were then introduced to replace raw query--page
memorization. The shared `P=32, epsilon=0.05` point was significantly above
pool-9 and statistically indistinguishable from full on both development
collections:

| Collection | tokens/full | pool-9 | probe residual | full | random residual |
|---|---:|---:|---:|---:|---:|
| HR | 11.94% | 0.4931 | 0.5065 | 0.5178 | 0.4934 |
| Finance-EN | 11.98% | 0.4503 | 0.4622 | 0.4732 | 0.4473 |

This point was frozen before opening a newly downloaded 1,360-page,
215-query Computer Science collection. The sealed result failed:

| Collection | tokens/full | pool-9 | probe residual | full | random residual |
|---|---:|---:|---:|---:|---:|
| Computer Science | 11.97% | 0.7027 | 0.7058 | 0.7170 | 0.7034 |

The nDCG@10 delta to pool-9 was only +0.0031 with 95% CI
[-0.0069, 0.0132]; the loss to full was -0.0112 with CI
[-0.0223, -0.0001]. Recall@100 lost 0.0073 to pool-9. The predeclared transfer
gate therefore rejects a three-domain fixed-probe claim.

Failure analysis also rejected easy rescues:

- raw residual witnesses reached only 0.7073 nDCG@10 at 11.50% tokens;
- increasing probe count to 64/128 reached 0.7080/0.7061 and harmed recall;
- a 32-probe scalar residual score sketch used only 11.09% of full bytes but
  fell below pool-9, both before and after qrel-free teacher scaling;
- pooled score overshoot was not the cause: full scores exceed pool-9 scores
  for essentially all page pairs in all three collections;
- mean 32-probe query-token distortion was 0.388 on Computer Science versus
  0.345 HR and 0.330 Finance, suggesting workload compressibility as a risk
  signal, not yet a validated selector.

## Next non-negotiable experiment

Do not build another residual variant. First reproduce the complete capacity
curves of Light-ColPali, ACL 2026 Prune-then-Merge, OmniColPress AGC and the
SIGIR 2026 Voronoi method on the public ViDoRe v3 domains. Generate full score
traces once and retain qrels only for final metrics. On those frozen surfaces,
test whether qrel-free workload geometry and representation distortion can
predict which operator/capacity is safe under query holdout and domain
transfer. The selector must include abstention to full and beat a single
global compression setting after charging index bytes, build time and MaxSim
latency. If it cannot, heterogeneous compilation is not the September paper
and the project should pivot to a modern compression operator or systems-only
artifact rather than continue local policy variants.
