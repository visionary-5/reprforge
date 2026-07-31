# Progressive Visual Index: Static Oracle Result

## Decision

The first allocation-headroom gate passes. This is evidence that a useful
progressive visual index can exist for this representation pair; it is not a
deployable online policy result.

On the full ViDoRe v3 HR English trace (318 queries, 1,110 pages), an
oracle-selected resident set of 25 pages (2.25%) already retains more than
95% of the full-visual nDCG@10 gain. The best tested set at or below the 30%
budget contains 150 pages (13.51%) and reaches 0.5436 nDCG@10, above both the
Markdown baseline (0.4947) and uniform full visual (0.5178).

## Corrected end-to-end baselines

| Policy | nDCG@10 | Build | Search | Total | Resident representation |
|---|---:|---:|---:|---:|---:|
| Markdown | 0.4947 | 32.91 s | 6.72 s | 39.63 s | 315 MB |
| Full visual | 0.5178 | 100.73 s | 8.01 s | 108.74 s | 585 MB |
| Tiered K=20 | 0.5009 | 32.71 s | 100.51 s | 133.23 s | 315 MB base + 491 MB peak cache |

The visual adapter originally serialized every decoded image to PNG and then
decoded it again. Removing that round trip cuts the full visual build from
276.7 to 100.7 seconds without changing any score. Therefore 3.06x, not
8.20x, is the representation build-time ratio used going forward.

K=20 remains a decisive negative baseline. It is already a naive form of
database cracking: locate text candidates, visualize every miss, and retain
the result forever. Over the trace it touches 932 pages (84%), makes 189
small visual-encoder calls, costs more time and combined memory than a single
full visual build, and loses retrieval quality. Query-driven construction
alone is not sufficient; candidate touch cannot be persistent admission.

## Oracle witness

The diagnostic selector ranks pages by their positive change in discounted
rank for qrel-labelled evidence when that page changes from text to visual.
It uses qrels and complete text/visual outcomes, so it is deliberately
unavailable to a runtime.

| Visual residents | Corpus fraction | nDCG@10 | Estimated batched build | Base + visual bytes |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0.4947 | 32.91 s | 315 MB |
| 25 | 2.25% | 0.5194 | 35.17 s | 328 MB |
| 100 | 9.01% | 0.5426 | 41.96 s | 368 MB |
| 150 | 13.51% | **0.5436** | 46.54 s | 394 MB |
| 333 | 30.00% | 0.5235 | 63.34 s | 491 MB |
| 1,110 visual only | 100% | 0.5178 | 100.73 s | 585 MB |

Only 249 of 613 relevant pages have a positive visual rank intervention.
Adding every visual page is not monotonic: some visual representations add
distractors or move evidence down the ranking. This explains why selective
visualization can beat uniform visual indexing rather than merely approach
it.

Two cheap full-stream orderings do not close the gap. Qrel frequency peaks at
0.4989 below 30% residency; text Top-20 frequency peaks at 0.5022. The useful
signal is not popularity by itself but the page-specific effect of changing
representation on ranking.

## Research consequence

Database cracking supplies the physical pattern--defer construction and
refine from queries--but does not supply the admission utility. Quake's
estimate--verify--commit discipline suggests how to avoid irreversible bad
actions. CLADO suggests measuring representation sensitivity through forward
interventions rather than assuming independent content labels. MURE is a
strong fixed compact visual representation, and LightSTAR covers transient
query-time candidate refinement. ReprForge's remaining system question is
therefore narrower and concrete:

> Can past, verified score interventions and reuse state estimate the future
> amortized value of keeping a page visual, so an online controller approaches
> the static oracle without qrels or future queries?

The next implementation is an estimate--verify--commit controller with three
separate actions: transiently refine a candidate, admit it to persistent
visual state only after positive evidence, and evict it when expected reuse
decays. It must be compared with text-only, uniform visual, K=20 retain-all,
LRU/LFU, a LightSTAR-style transient-only policy, and the static oracle on
chronological train/validation/test query prefixes. If a held-out online
policy cannot retain at least 95% of visual gain while remaining below 30%
residency and beating full visual end-to-end at the registered horizon, the
persistent-admission branch stops.

## Reproducibility boundary

The committed summary contains metrics, costs, and SHA-256 digests. Full
318x1,110 score matrices, images, models, and machine-specific logs remain in
the private experiment workspace because they are generated artifacts. The
replay implementation is `reprforge/progressive_oracle.py`, and the frozen
output is `results/vidore-progressive-oracle/summary.json`.
