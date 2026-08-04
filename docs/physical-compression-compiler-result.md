# Physical compression compiler — development result

Date: 2026-08-04. Status: development candidate, frozen before any
physical-method evaluation on Industrial, Pharmaceuticals, or the sealed
French collections. This is stronger than the configuration selector but is
not yet a paper claim.

## Method candidate

The compiled index has two physically charged views:

1. a semantic hierarchical pool-4 cover for every document;
2. full ColPali vectors for a static subset of document anchors.

The subset is compiled from unlabeled fit-query full/pool-4 rankings. For each
document, the compiler measures recurrent Top-10 membership flips and smooth
rank displacement within the competitive Top-100. It independently orders
documents by each event utility per full-anchor byte and deterministically
round-robins the two orders. This parameter-free boundary coreset avoids a
scalar-weight pathology observed in development: discrete flips otherwise
dominate continuous displacement for every tested weight.

At query time, the pool-4 cover scores every document and full MaxSim scores
the stored anchors. A ridge-affine model with fixed `ridge=1e-3` predicts the
query-local full-minus-pool residual from paired anchor scores; predictions
are clipped to the observed anchor residual range, and anchor documents keep
their exact full scores. The cheap view remains resident for anchors, so its
bytes and token work are charged rather than treated as a free fallback.

The development operating point stores full anchors up to 65% of full-bank
document bytes. Together with the 24.95% pool-4 cover, measured resident
fractions are 89.91--89.95%; small variation comes from byte-budget packing.

## Held-out development result

Queries use the frozen stable query-ID hash split: two thirds compile the
static document plan and one third evaluate future workload recurrence. Qrels
are never compiler features. All safety intervals below use 4,000 paired
query bootstrap samples.

| Domain | Full anchors | resident / full | Top-10 overlap | nDCG@10 mean regret | nDCG@10 upper regret | Recall@100 upper regret | worst-5% CVaR | safety |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HR | 721 / 1,110 | 89.91% | .9835 | -.00080 | .00738 | .00752 | .1038 | pass |
| FinanceEN | 1,912 / 2,942 | 89.94% | .9798 | .00223 | .00619 | .00085 | .0694 | pass |
| Computer Science | 884 / 1,360 | 89.95% | .9649 | .00254 | .00829 | .00195 | .1033 | pass |

The qrel-free ranking certificate also passes on all three held-out splits;
its one-sided lower Top-10 overlap values are .9765, .9725, and .9519, and the
lower full-Top-10 retention in candidate Top-100 is 1.0 throughout. This does
not make the certificate distribution-free.

Before freezing, the compiler was also converted from a convention into a
three-fold qrel-free cross-certificate over the outer-fit workload. Aggregate
cross-fit Top-10 overlap is .9660 on HR, .9790 on FinanceEN, and .9645 on
Computer Science; every individual fold and all three aggregate certificates
pass. The final plan is refit on all outer-fit queries only after that
certificate passes. This produces the same 721/1,912/884 anchor counts as the
development points above, while reserving every outer evaluation query from
both planning and abstention.

The best fixed state that passes all three development domains remains full
at 100%. At this point the physical compiler saves about 10.1% persistent
document-vector bytes. The gain is modest but is a real physical frontier
improvement, not configuration-level dataset routing.

## Mechanism controls

The calibration component is necessary. At the same allocation, replacing
pool scores with raw full scores for selected documents fails the safety gate
on all three domains. Quality can initially become worse as more full
documents are added because MaxSim scores from different token capacities are
not automatically comparable.

The allocation component is also necessary. A deterministic random anchor
allocation with the same dual-view calibration fails HR and FinanceEN at the
same byte point. Computer Science alone is insufficient to distinguish it,
which is why collection-macro and worst-collection results are required.

More capacity is not monotonically safer: the same boundary coreset at a 70%
full-anchor budget passes HR and FinanceEN but fails Computer Science with an
nDCG@10 upper regret of .01122. A compiler therefore needs a qrel-free
selection certificate or abstention; it cannot assume that adding
heterogeneous raw capacity monotonically approaches full behavior.

## Physical materialization check

The Computer Science point is materialized as two real fp16 banks: the full
pool-4 cover plus an 884-document full-anchor bank. The measured document
vectors occupy 322,570,240 bytes versus 358,604,800 bytes for full, exactly
89.951%. Scoring the anchor bank on A100 took 2.218 seconds for 215 queries and
884 anchors in the current unoptimized replay.

The materialized anchor scores match the earlier direct full surface with
maximum absolute difference `1.91e-6` and mean absolute difference
`5.02e-11`. Composing only the materialized cheap and anchor surfaces exactly
reproduces the development result: nDCG@10 upper regret .00829 and Recall@100
upper regret .00195. The combined bank is therefore physically realizable;
the current latency is still a single replay, not an optimized systems claim.

## Remaining gates

Before this becomes the paper method:

1. write the plan, materialize the bank, and issue a qrel-free certificate in
   separate processes before label evaluation on every new collection;
2. pass physical-method validation and at least two sealed French transfers;
3. reproduce modern homogeneous Light/AGC/Prune-then-Merge curves and show
   better safe byte--quality area, not only improvement over hierarchical
   pool-4/full;
4. measure warm/cold latency, token work, memory, build cost, and update
   amplification for the actual dual-view execution.

The relevant compact artifacts are
`results/diagnostics/physical-compression-{hr,finance,cs}-v5-pareto.json`,
`physical-compression-cs-v6-materialized.json`, and
`results/systems/cs-dual-view-anchor-bank-v1.json`.

## Frozen physical validation result

The frozen compiler was subsequently run on Industrial and Pharmaceuticals
without parameter changes. In both collections its three-fold internal
certificate passed before materialization; the final plans retained 3,408 and
1,503 full anchors. The actual dual-view banks occupied 89.940% and 89.932% of
full document-vector bytes. Reserved-query runtime certificates were written
and hashed before physical-method relevance evaluation, and both passed with
Top-10 overlap .9719/.9736 and one-sided lower bounds .9615/.9659.

The relevance safety gate nevertheless failed narrowly on both collections:

| Domain | resident / full | mean nDCG@10 regret | nDCG@10 upper regret | Recall@100 upper regret | qrel-free certificate | relevance safety |
|---|---:|---:|---:|---:|---|---|
| Industrial | 89.940% | .00312 | .01086 | .00000 | pass | **fail** |
| Pharmaceuticals | 89.932% | .00558 | .01104 | .00111 | pass | **fail** |

Anchor score parity is within `1.91e-6`, so this is not a materialization or
replay discrepancy. It is a 2/2 physical-state false-safe result for the
current ranking certificate. The frozen boundary-coreset candidate is
therefore rejected as the paper method even though it improves mean regret
and comes close to the declared tolerance. Thresholds are not retuned on
these validation outcomes.

The failure initially redirected the method toward workload residual matrix
completion: select full-document anchor columns from the qrel-free
full-minus-pool score matrix, then reconstruct a future query's complete
residual row from physically observed anchor residuals. That branch has now
also been bounded and rejected.

## Residual-matrix branch result

The standardized, row-centered fit residual is not low rank at a useful
physical budget. Rank 64 explains 72.3--76.5% of residual energy on HR,
FinanceEN, and Computer Science, but only 65.7% on Industrial. Rank 128 is
needed for 88.4--98.0% across the five opened collections. Entropy effective
ranks range from 58.5 to 101.9. A universal rank-16 residual model is therefore
not supported by the measured surfaces.

Ordinary pivoted-column ridge completion also fails as a ranking method. With
a fixed pool-Top-100 candidate stage, 128 full anchor columns happen to pass
the mean safety gate on HR at 36.5% of full bytes, but the same setting fails
FinanceEN and Computer Science. Increasing to 256/512 anchors and candidate
depth 200 does not produce a cross-domain safe trend. On HR, a truncated-SVD
model at rank 64 still has nDCG@10 upper regrets .02321 and .02145 with
128/256 anchors, both above the .01 tolerance. This is enough to stop the
ordinary CUR/SVD branch; no additional GPU sweep is justified.

The lesson is structural. Frobenius residual reconstruction, global ridge
error, and qrel-free Top-k agreement can all improve while the few relevance-
bearing rank swaps that determine nDCG remain unresolved. The next protocol
will treat relevance calibration as a deliberately scarce compilation input,
not pretend that a stronger unlabeled score proxy is automatically a safety
certificate. The five opened collections may be used for this method
development; the three French collections remain untouched for transfer.
