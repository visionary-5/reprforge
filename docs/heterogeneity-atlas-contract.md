# Heterogeneity Atlas v0 contract

## Question

Before committing to another controller, the atlas asks where the useful
heterogeneity actually lives:

1. **Global representation:** one representation dominates most queries.
2. **Query route:** different queries prefer different whole indexes.
3. **Static document plan:** a document can keep one route across queries.
4. **Query-document interaction:** relevant evidence appears only when routes
   can be combined at interaction time.

The atlas is a diagnostic substrate, not the claimed paper contribution.

## Frozen inputs

Every route supplies a complete `query x corpus` score surface with aligned
identifiers. Relevance is stored separately. The first run uses existing HR,
Finance, and biomedical-interaction traces; it does not re-encode a corpus.
Cost metadata is reported but never used to normalize retrieval scores.

## Comparability rule

Raw scores from heterogeneous retrievers are not assumed comparable. Uniform
routes use their original ranking. Any mixed per-document plan uses
within-query rank percentiles, with the best-global percentile used only as a
bounded secondary tie-breaker. Label-free portfolios use reciprocal-rank
fusion or best-rank pooling.

## Nested evidence levels

- **Best global:** selected on fit queries and evaluated on held-out queries.
- **Label-free portfolio:** all indexes are resident; RRF/best-rank pooling is
  a coherent retrieval ranking and uses no qrels.
- **Query-route oracle:** selects the best whole-index ranking separately for
  each held-out query. It uses held-out labels and is only an upper bound.
- **Static document diagnostic:** for each document observed as relevant on
  fit queries, select the route with its best mean relevant rank. Freeze the
  route per document and evaluate on held-out queries. This uses fit labels
  and is not presented as a learned deployable compiler.
- **Relevant-evidence best rank:** asks whether each relevant item appears in
  any route's top-k. This is not a coherent ranking; it is an evidence
  availability ceiling.

## Interpretation

- Static document plan materially exceeds best-global on holdout: prioritize
  an offline heterogeneous index compiler and learn deployable document-side
  features.
- Query oracle is large while the static plan is flat: prioritize query-side
  index routing.
- Only best-rank evidence is large: the unit of heterogeneity is likely the
  query-document interaction; test active acquisition or late fusion.
- Label-free portfolios close most of the gap: a new compiler needs a cost or
  lifecycle advantage over simple multi-index fusion, not only quality.
- All gaps are small: reject the premise on this representation family and
  change the action space rather than adding controller complexity.

## Leakage boundary

All label-using fields live under `diagnostic_upper_bounds` or carry an
explicit warning. They may motivate a method but cannot be reported as its
deployable test result. Final claims require a frozen train/dev/test protocol,
strong full-index and fusion baselines, multiple public collections, paired
confidence intervals, and measured build/storage/query cost.
