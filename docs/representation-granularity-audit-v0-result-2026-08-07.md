# Representation Granularity Audit v0

## Outcome

The preregistered region-headroom gate did **not** pass. Deterministically
splitting selected pages into four visual units did not consistently preserve
whole-page retrieval quality while reducing negative-page interference, and it
expanded the physical Omni index by approximately four times.

This is a useful negative result. The earlier page-value atlas showed that the
value of an expensive visual representation is sparse and signed. This audit
shows that the sparsity cannot be converted into a useful index merely by
materializing every mechanical sub-page crop. Granularity, representation, and
admission must be compiled jointly.

## Frozen setup

For each of Pharmaceuticals and Industrial, the frozen cohort contains:

- 24 pages with the largest positive singleton Omni value;
- 24 pages with the largest negative singleton Omni value;
- 24 deterministically sampled neutral pages.

The same 72 parent pages and all domain queries are used in each organization:

1. one complete-page visual unit;
2. four overlapping fixed quadrants;
3. up to four deterministic whitespace-aware XY-cut regions.

All units use the same Full OmniColPress representation. Region scores are
aggregated to the parent page with `max` before parent-level evaluation. This
prevents a page from gaining multiple retrieval slots merely because it was
split into several units. Only queries with at least one relevant parent in the
frozen 72-page cohort are evaluated.

The preregistered gate required a region organization to retain at least 95% of
complete-page nDCG and reduce irrelevant Top-10 exposure of negative-value pages
by at least 10%.

## Results

### Pharmaceuticals (85 eligible queries)

| Organization | Units | Parent nDCG@10 | Hit@10 | Negative irrelevant exposure | Build wall time | Physical index |
|---|---:|---:|---:|---:|---:|---:|
| Complete page | 72 | 0.764825 | 0.917647 | 473 | 54.47 s | 0.840 GB |
| Fixed quadrants | 288 | 0.765633 | 0.917647 | 466 (-1.48%) | 90.70 s | 3.231 GB |
| XY-cut regions | 273 | 0.774781 | 0.929412 | 460 (-2.75%) | 89.18 s | 3.181 GB |

XY-cut gives a small positive quality diagnostic: `+0.009956` nDCG and one
additional hit among 85 eligible queries. It does not approach the required
interference reduction and costs 3.79 times as many physical index bytes per
parent.

### Industrial (88 eligible queries)

| Organization | Units | Parent nDCG@10 | Hit@10 | Negative irrelevant exposure | Build wall time | Physical index |
|---|---:|---:|---:|---:|---:|---:|
| Complete page | 72 | 0.788349 | 0.954545 | 472 | 53.85 s | 0.830 GB |
| Fixed quadrants | 288 | 0.763660 | 0.954545 | 453 (-4.03%) | 97.07 s | 3.273 GB |
| XY-cut regions | 288 | 0.749266 | 0.943182 | 473 (+0.21%) | 90.16 s | 3.346 GB |

Fixed quadrants remove a few negative-page exposures but lose `0.024689` nDCG.
XY-cut loses `0.0390837` nDCG, loses one eligible-query hit, and slightly
increases negative-page exposure. Neither organization passes the gate.

## Physical explanation

The tested Full Omni encoder produces a fixed `1,422 x 2,048` representation
for every input image in this setup. Cropping a page does not yield a
proportionally smaller persistent representation. Encoding four crops therefore
creates approximately four times the vectors and four times the stored bytes.

Measured build wall time rises by only 1.64--1.80 times because model loading,
profiling, and index serialization contain fixed costs. Storage is the more
faithful steady-state signal here: the sub-page indexes are 3.79--4.03 times the
complete-page indexes.

## Research interpretation

The result rejects the weak method:

> split every page into regions, independently embed every region, and retrieve
> their parent by maximum score.

It does not reject multi-granularity knowledge-base compilation. Instead it
narrows the necessary method:

1. **Structure before representation.** Title, body, table, figure, caption,
   axis, header, and footer must be represented as typed objects and relations,
   not as an undifferentiated set of image crops.
2. **Selective representation allocation.** Most text-native blocks should not
   pay for a high-fidelity visual vector. Only units whose visual state adds net
   workload value should be promoted.
3. **Signed admission.** A unit can help one query and harm several others.
   Admission must estimate both evidence gain and cross-query interference.
4. **Parent-aware retrieval.** Several representations of the same logical
   object must be calibrated and aggregated before competing with other pages;
   otherwise greater fragmentation itself changes ranking.
5. **Variable-cost operators.** A useful region plan needs compact or
   variable-token representations. Fixed-token Full Omni per crop cannot be the
   physical implementation of a cost-saving region tier.

The resulting build-time decision is not simply `page -> VLM or no VLM`. It is:

```text
(logical knowledge unit, structural context)
    -> choose unit boundary
    -> choose text / compact visual / high-fidelity visual representation
    -> choose absent / transient / persistent state
    -> aggregate through its page and document parents.
```

## Next gate

The current ViDoRe exports provide page images and page OCR/Markdown but no
faithful block boxes or original document hierarchy. A real structure-aware
method cannot be validated from these artifacts alone.

The next experiment should therefore add one structure-rich document corpus and
freeze a typed document graph with at least text blocks, tables, figures, and
captions. It should compare:

- page-only text and Full visual baselines;
- text blocks plus page visual representation;
- typed objects with visual representation only for selected tables/figures or
  low-confidence text regions;
- an oracle typed-object allocation under equal build bytes;
- a realizable signed-value selector using only ingestion-time structure and a
  historical workload split.

The continuation gate is whether typed selective allocation improves the
quality--bytes frontier over both complete pages and all-region materialization
in at least two domains. If it does not, ReprForge should retain page-level
visual units and focus on representation choice and lifecycle rather than claim
sub-page visual compilation.

## Artifacts

- `configs/representation-granularity-audit-v0.json`
- `results/compiler-feasibility/representation-granularity-audit-v0/pharmaceuticals.json`
- `results/compiler-feasibility/representation-granularity-audit-v0/industrial.json`

Raw images, embeddings, rankings, logs, timing files, and manifests remain on
the AutoDL data disk under `granularity-audit-v0-*`. The two checked-in summaries
were SHA-256 verified after transfer.
