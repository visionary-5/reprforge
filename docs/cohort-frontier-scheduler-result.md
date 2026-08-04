# Cohort-Frontier Scheduler Result

Date: 2026-08-04. Status: HR complete; Finance frontier complete and strong
baseline recovery pending.

## Decision so far

The qrel-free cohort-frontier scheduler passes its frozen work gate on HR and
Finance-EN and its real A100 gate on HR.  It also transfers from exact page
work to real A100 time on Finance-EN.  The remaining Finance
static-popularity run must be recovered before calling the two-domain system
comparison complete.

This is the first ReprForge mechanism in the current branch that improves the
trajectory without changing K, fusion, final cohort coverage, or the model.
It is a scheduling result, not yet a complete paper claim: the current replay
allows unrestricted reordering of a queued benchmark workload.

## Exact construction-work replay

Every schedule eventually constructs the same candidate-union pages.  The
replay charges the exact cumulative number of unique visual pages at each
atomic batch publication.

| Dataset | Schedule | Mean pages at query completion | P50 | P95 | nDCG--work AUC |
|---|---|---:|---:|---:|---:|
| HR | FIFO | 718.6 | 768 | 892 | 0.50276 |
| HR | Static popularity | 593.5 | 645 | 865 | **0.51189** |
| HR | Cohort frontier | **579.8** | **627** | **855** | 0.51012 |
| Finance-EN | FIFO | 1,208.7 | 1,341 | 1,811 | 0.54806 |
| Finance-EN | Static popularity | 893.8 | 891 | **1,684** | 0.55301 |
| Finance-EN | Cohort frontier | **861.4** | **824** | 1,698 | **0.55355** |

Against FIFO, frontier reduces mean completion work by 19.3% on HR and 28.7%
on Finance-EN.  It is also better than static popularity on the label-free
completion-work objective.  Static popularity has slightly higher post-hoc
quality AUC on HR, which is why it is retained as a real GPU baseline rather
than dismissed after the work replay.

## HR A100 result

All executions use one A100-SXM4-80GB, ColPali-v1.1, image batch four, scoring
batch 16, request batch eight, BM25 Top-20 and unbounded resident state.

| Schedule | Final pages | End-to-end | First batch | Batch P95 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| FIFO | 895 | 98.03 s | 15.54 s | 9.04 s | 0.53729 |
| Static popularity | 895 | 99.26 s | 12.26 s | 4.66 s | 0.53705 |
| **Cohort frontier** | 895 | **91.85 s** | **6.73 s** | **4.52 s** | 0.53648 |

The frontier is 6.3% faster end-to-end than FIFO and 7.5% faster than static
popularity.  It cuts first publication latency by 56.7% versus FIFO and 45.1%
versus popularity.  Frozen-score quality--time replay over these real batch
traces gives:

| Schedule | Mean nDCG over full-build horizon | Gain vs fair full prebuild | Time to 90% final fusion gain | Evidence P50 |
|---|---:|---:|---:|---:|
| FIFO | 0.49841 | +0.01340 | 98.02 s | 82.01 s |
| Static popularity | 0.50708 | +0.02199 | 94.31 s | 72.98 s |
| **Cohort frontier** | **0.50792** | **+0.02277** | **86.67 s** | **64.25 s** |

Thus the scheduling win is not obtained merely by publishing low-quality
queries early.  Frontier is better than the strongest simple schedule on the
time-integrated metric even though static popularity won the hardware-free HR
work AUC.

## Finance-EN transfer available so far

| Schedule | Final pages | End-to-end | First batch | Batch P95 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 1,855 | 190.50 s | 15.98 s | 9.13 s | 0.56280 | 0.88045 |
| **Cohort frontier** | 1,855 | **172.85 s** | **4.85 s** | **8.02 s** | 0.56255 | 0.88045 |
| Static popularity | pending | pending | pending | pending | pending | pending |

Frontier is 9.3% faster end-to-end and cuts first publication latency by
69.6%.  The observed nDCG difference is -0.00026 and Recall@100 is unchanged.

## Reproducibility correction

The frozen score surface proves that final ranking semantics and candidate
coverage are order invariant.  Real bf16 image encoding is not bitwise
invariant to the changed image-call packing: HR nDCG@10 spans 0.00081 across
the three schedules even though all construct the same 895 pages.  The same
effect was already observed across request batch sizes, and canonical padding
did not remove it.  Therefore report frozen semantic parity and real-model
numerical variation separately; do not claim bitwise-identical final ranks.

## What the result contributes—and what it does not

The mechanism is a candidate for the paper only as part of an anytime index
compiler:

> A cheap locator exposes a query--representation dependency graph.  Scheduling
> atomic visual cohorts on the resident frontier reduces the construction work
> and wall-clock time required to publish useful evidence, while preserving
> the final physical state.

Deferred visual ingestion itself is covered by DVI.  Query grouping and
locality scheduling are covered by systems such as CaGR-RAG.  ReprForge must
therefore establish the narrower combination of expensive physical
multi-vector construction, reusable versioned state, and evidence-quality
progress under construction work.  The next experiment must impose bounded
arrival windows or replay a public workload; unrestricted full-stream
lookahead is an optimistic batch setting, not an online serving result.

Revision safety also remains open.  On both frozen domains, approximately
19--22% of per-query BM25-to-fusion revisions reduce nDCG@10, with fifth
percentile losses of -0.116 on HR and -0.183 on Finance-EN.  Scheduling makes
useful state available earlier but does not decide whether a completed
revision is safe to publish.  Answer-level correctness and answer
stabilization are required before submission.
