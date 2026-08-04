# Compression-risk baseline — semantic hierarchical-pooling capacity curve

Date: 2026-08-04. This is the first result under
`docs/compression-risk-contract.md`. It is a baseline and phenomenon result,
not a proposed-method result.

## Pool-9 phenomenon result

Semantic hierarchical pool-9 retains 11.068% of the full document vectors and
exhaustive
MaxSim document-token work. It does not pass the frozen safety gate on any of
the three opened domains.

| Domain | Full / pool-9 nDCG@10 | Mean regret | Worst-5% CVaR | Catastrophic queries | Full Top-10 retained in pool Top-100 | Top-10 overlap | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| HR | .5178 / .4931 | .0246 | .3848 | 21.7% | 99.843% | 76.38% | fail |
| FinanceEN | .4732 / .4503 | .0228 | .4138 | 15.5% | 99.871% | 76.83% | fail |
| Computer Science | .7170 / .7027 | .0143 | .2590 | 16.3% | 99.907% | 81.63% | fail |
| macro | — | .0206 | .3525 | 17.8% | 99.873% | 78.28% | 0/3 pass |

Positive regret means pool-9 is worse than full. A catastrophic query has
per-query nDCG@10 regret greater than 0.10. CVaR is the mean regret among the
worst 5% of queries. The gate requires the one-sided bootstrap upper mean
nDCG@10 and Recall@100 regrets to both be at most 0.01.

The corresponding nDCG@10 upper regrets are .0373, .0356, and .0250. Recall is
not the main failure on HR or Computer Science: pool-9 Recall@100 is slightly
higher than full on both. FinanceEN fails both nDCG and Recall upper bounds.

## Interpretation

The new metrics expose a stable two-stage structure:

1. aggressive compression is an excellent **candidate locator**: across the
   three domains, only 0.127% of full Top-10 items escape pool-9 Top-100;
2. it is not a safe final ranker: only 78.28% of full Top-10 membership is
   preserved at Top-10, and the tail quality loss is much larger than the mean.

This rules out a paper claim that pool-9 itself is nearly lossless. It also
changes the optimization target. The compiler should spend capacity on the
uncertain ordering boundary inside a high-recall compressed cohort, while
retaining the compressed cover for candidate generation. A global score-error
regressor is poorly aligned with this target; boundary inversions and their
cross-workload stability are the relevant qrel-free events.

The finding is stronger than the earlier residual-token story because it is
consistent on all three opened domains and does not depend on the residual
selector. It does not yet demonstrate that boundary risk is predictable.

## Relation to current compression work

Light-ColPali reports that offline patch importance is highly query-dependent
and therefore favors merging over pruning. Prune-then-Merge makes a stronger
homogeneous operator by adaptively pruning low-information patches before
hierarchical merging. OmniColPress/AGC learns universal query tokens, selects
salient centroids from their final-layer attention, and assigns all document
tokens to those centroids under a fixed vector budget.

Those papers occupy the “better homogeneous compression operator” claim. The
unoccupied ReprForge question is operator-agnostic compression safety: predict
where any of these operators will disturb the competitive ranking boundary,
choose capacity at a real physical decision unit, and abstain when the signal
does not transfer.

Source inspection of OmniColPress commit
`4a559677bbc8a3ea0c10322a721b52bb70d382ec` confirms that AGC is not a
drop-in post-processing step for the existing ColPali bank. Its saliency comes
from learned Universal Query tokens appended during model training and their
last-layer attention. The immediately reusable baseline from that repository
is non-parametric hierarchical clustering; faithful AGC requires its published
checkpoint/model path or training recipe and must be costed separately.

## Next experiment

Before opening validation domains:

1. retain the semantic pool-25/9/4/full ladder and add a 2-D spatial control,
   Prune-then-Merge, and a faithful fine-tuned Light/AGC curve where model
   compatibility permits;
2. define qrel-free boundary events from full-reference fit rankings: which
   pool Top-100 pairs invert around cutoffs 10 and 100;
3. measure event recurrence under query-hash cross-fit and fit a calibrated
   upper-risk predictor from workload frequency, margin, and operator
   disagreement;
4. build actual hybrid indexes at increasing physical-unit coverage and score
   the end-to-end rankings;
5. compare safe coverage with fixed rates, random matched allocation, and an
   oracle ordering before touching Industrial or Pharmaceuticals.

If boundary events do not recur or the calibrated upper bound fails on the
Computer Science development domain, stop this method branch before scaling.

## Artifacts

- `results/compression-risk/hr-pool9-v1.json`
- `results/compression-risk/finance-en-pool9-v1.json`
- `results/compression-risk/computer-science-pool9-v1.json`

Each artifact contains per-query relevance metrics, rank fidelity arrays,
bootstrap intervals, input hashes, the decision-unit declaration, and physical
byte/token-work ratios.

## Capacity-curve update

The frozen pool-4 and pool-25 points were subsequently generated from the same
full banks. All points are complete-corpus score surfaces; no qrels enter
pooling or scoring.

| Domain | Pool | vectors / full | nDCG@10 | mean regret | one-sided upper regret | worst-5% CVaR | full Top-10 in candidate Top-100 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HR | 25 | 3.98% | .4591 | .0587 | .0737 | .5310 | 98.145% | fail |
| HR | 9 | 11.07% | .4931 | .0246 | .0373 | .3848 | 99.843% | fail |
| HR | 4 | 24.95% | .5130 | .0047 | .0137 | .2604 | 100% | fail |
| FinanceEN | 25 | 3.98% | .4416 | .0316 | .0471 | .5097 | 98.770% | fail |
| FinanceEN | 9 | 11.07% | .4503 | .0228 | .0356 | .4138 | 99.871% | fail |
| FinanceEN | 4 | 24.95% | .4671 | .0060 | .0140 | .2453 | 100% | fail |
| Computer Science | 25 | 3.98% | .6812 | .0358 | .0501 | .4035 | 99.302% | fail |
| Computer Science | 9 | 11.07% | .7027 | .0143 | .0250 | .2590 | 99.907% | fail |
| Computer Science | 4 | 24.95% | .7207 | -.0037 | .0038 | .1677 | 100% | **pass** |

Pool-4 is the useful compiler base. Its mean nDCG@10 regret is below .01 on all
three domains and it improves over full on Computer Science, but uncertainty
keeps HR and FinanceEN just outside the .01 certificate. The best fixed point
that passes all three development domains under the frozen gate is therefore
still full. A heterogeneous method has a concrete, narrow target: retain the
75% pool-4 byte reduction while upgrading enough genuinely risky physical
units to move the HR/FinanceEN upper bounds below .01.

Tail harm remains material even when the mean is safe. Pool-4 worst-5% CVaR is
.168--.260 and 100% candidate retention does not imply stable final ranking.
This confirms that the method must optimize boundary ordering rather than only
Top-100 candidate coverage.

### Development-only ranking assurance result

The qrel-free gate frozen in the contract accepts only Computer Science
pool-4 among the nine compressed configurations. That exactly matches the
qrel-based safety decisions on these nine development points:

| Domain | cheapest accepted state | resident-vector fraction | qrel-free gate | qrel safety gate |
|---|---|---:|---|---|
| HR | full | 100% | abstain | pass by identity |
| FinanceEN | full | 100% | abstain | pass by identity |
| Computer Science | semantic hierarchical pool-4 | 24.95% | pass | pass |

The resulting macro resident-vector fraction is 74.98%, versus 100% for the
only fixed state that is safe on all three collections. This is encouraging
but not held-out evidence: the thresholds were formulated after inspecting
these development domains. The method becomes credible only if the unchanged
gate selects safe rates on Industrial and Pharmaceuticals. It also remains a
configuration selector, not yet the final physical-unit heterogeneous
compiler.

### First held-out validation: Pharmaceuticals

The frozen ranking gate transferred without threshold changes on the first
validation collection. Three qrel-free certificates were written and hashed
before the qrel Parquet was opened. The gate accepted pool-4 and abstained on
pool-9 and pool-25; the later relevance evaluation found exactly the same
safety decisions.

| state | resident fraction | Top-10 overlap | qrel-free gate | nDCG@10 | mean regret | one-sided upper regret | qrel safety |
|---|---:|---:|---|---:|---:|---:|---|
| pool-4 | 24.95% | .9137 | pass | .5743 | -.00013 | .00648 | pass |
| pool-9 | 11.07% | .8442 | abstain | .5468 | .02727 | .03665 | fail |
| pool-25 | 3.98% | .7984 | abstain | .5531 | .02106 | .03305 | fail |

Pool-4 also passes the Recall@100 gate with a one-sided upper regret of
.00329, while reducing measured resident bytes and exhaustive document-token
work by 75.05%. Its worst-5% nDCG@10 CVaR remains .1719, so the result supports
the declared mean-safety claim but does not erase query-level tail harm.

### Second held-out validation: Industrial

The same frozen gate was then applied to Industrial, again with three
qrel-free certificates written and hashed before the qrel Parquet was opened.
It abstained on all compressed states. Post-certificate relevance evaluation
confirmed that all three states violate both parts of the frozen safety gate.

| state | resident fraction | Top-10 overlap | qrel-free gate | nDCG@10 | mean regret | one-sided upper regret | Recall@100 upper regret | qrel safety |
|---|---:|---:|---|---:|---:|---:|---:|---|
| pool-4 | 24.95% | .8629 | abstain | .4481 | .01063 | .01806 | .02119 | fail |
| pool-9 | 11.07% | .7562 | abstain | .4378 | .02092 | .03200 | .03463 | fail |
| pool-25 | 3.98% | .6898 | abstain | .4130 | .04573 | .06024 | .04323 | fail |

Industrial is the useful counter-case: pool-4 retains every full Top-10 item
inside candidate Top-100, yet final ordering is not safe. Its worst-5%
nDCG@10 CVaR is .2372. This rules out candidate retention alone as the risk
criterion and justifies the overlap/boundary component of the certificate.

### Configuration-level validation summary

Across the two held-out collections and six compressed configurations, the
unchanged qrel-free gate matches all six post-certificate safety decisions:
zero false-safe and zero false-reject decisions. It selects pool-4 on
Pharmaceuticals and full on Industrial, giving a 62.48% macro resident-vector
fraction and a macro selected mean nDCG@10 regret of -.00007. The best single
fixed state that is safe on both collections is full at 100%, so selection
saves 37.52 percentage points of resident capacity. At the evaluated ladder
points, its selection also matches the qrel-aware oracle-safe selection.

This establishes the required configuration-level transfer sanity check; it
does not establish the paper method. The decision unit is still an entire
collection, and the certificate consumes full-reference rankings for the
same unlabeled workload. The next milestone is to compile and materialize a
document/shard-level hybrid on fit queries, evaluate it on future held-out
queries, and beat the best fixed modern compression operator. Industrial is
the primary development target because global pool-4 is unsafe there while a
physical compiler may retain full capacity only for risky units.

The operator is `colpali-engine==0.3.12` `HierarchicalTokenPooler`, source
SHA-256 `db24f2f1d381674c8f3f8ffc050696a3085151cfc8def44e642761d24c9cb891`.
It constructs an all-token similarity matrix, applies Ward hierarchical
clustering, averages each cluster, and L2-normalizes the result. This is a
training-free semantic-merging baseline, not the fine-tuned Light-ColPali
model; the fine-tuned result remains pending.

The current replay scorer pads small batches and casts vectors to float32. Its
single-pass timings are not an optimized latency comparison and are excluded
from the claim. `resident-index` bytes and document vectors per exhaustive
query are explicit in the JSON; optimized warm/cold latency remains required.

Additional artifacts are `hr-pool4-v1.json`, `hr-pool25-v1.json`, and the
corresponding FinanceEN and Computer Science files under
`results/compression-risk/`. Industrial and Pharmaceuticals validation
evaluations, pre-qrel certificates, post-certificate label manifests, and the
cross-collection `validation-gate-transfer-v1.json` summary are stored in the
same directory. The derived banks and raw score surfaces are kept
under `/data/ldf/reprforge/vidore-official-a70f23a/results/
compression-ladder-20260804` and `compression-risk-validation-20260804` on
`a100-server`.
