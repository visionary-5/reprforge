# Evidence index

Use [claim-ledger.md](claim-ledger.md) for interpretation and
[provenance.json](provenance.json) for exact file hashes. Original numerical
outputs are packaged in `evidence/historical-evidence.tar.gz` through Git LFS.
After [verified extraction](../experiments/README.md), the paths in this table
exist below your chosen evidence directory. Code directories have the same names
under `experiments/`. `tables/` contains historical reports, not new evidence.

| Paper question / number | Experiment directory | Raw output / fields | Important limit |
|---|---|---|---|
| Locality of 19 inspected releases | `2026-09-04-public-release-stage-census-v1` | `raw-output/census.json`; analysis `transitions.json` | Tensor names/config metadata; analyzer's legal-cut label omits processor condition. |
| Vision-adapter negative example | `2026-08-22-public-release-dependency-trace-v1` | `result.json` | Turkish ColPali vision LoRA changes invalidate this cut. |
| Processor equivalence over 2,225 pages | `2026-08-22-processor-equivalence-certificate-v1` | `result.json` | Collection-scoped observed output equality; no universal certificate. |
| Independent exact official ColQwen2 upgrade; 128 pages | `2026-08-22-colqwen2-release-transition-v1` | `cpu-result.json`, `gpu-result.json` | Independent target load, persisted IR, matched batch-4 Energy slice. |
| Existing ColQwen2.5 endpoint examples | `2026-08-21-incremental-index-recompile-v1`, `2026-08-21-vietnamese-domain-adapter-replay-v1` | `result-v1.json` / `result.json` | Vietnamese run reuses the earlier state; same 128-page slice. Runner shared with incremental experiment. |
| Pooled main table, 3,420 pages / 1,033 queries | `2026-09-07-pooled-gallery-upgrade-matrix-v1` | `raw-output/output-{12845056,602112}/{result,rankings,per_query}.json`, `raw-output/query_mask.json` | Use corrected analysis, not original aggregate; remove Shift queries, keep its pages. BF16 exists only for v0.2 and Metric-AI. |
| Ordered-sensitive metrics and clustered intervals | Same pooled directory | `extras.py` applied to saved rankings/per-query rows | RBO/Kendall are not ordered top-k equality. |
| Second lineage mixed-pool comparison | `2026-09-09-colqwen2-pooled-upgrade-v1` | `raw-output/output/result.json`, rankings and per-query files | 602,112 pixels. 40-page tensor check uses shared prefix, unlike the older independent Energy endpoint. |
| Codec frontier and stage shares | `2026-09-04-codec-frontier-stage-anatomy-v1` | `raw-output/` results; `analysis-output/{anatomy-table,leverage-table}.md` | BF16 consistency vs lossy codecs. ColQwen2 frontier used 12.8M, despite an old protocol assumption of 602k. |
| 73.4% complete migration | `2026-08-22-mmdocir-version-transition-v1` | `result.json`, `replay-result.json`, `provenance.json` | Approximate PCA-256; prior-run frozen raw baseline. |
| 77.8–85.2% RTX 4090 build saving | `2026-08-24-paper4090-main-table-repeats-v1` | `raw-output/*-run{1,2,3}.json`, host manifest | Technical repeats, approximate routes, excluded calibration/loading/query costs. |
| New independent 3B target checks | `independent-endpoint` | `independent-endpoint-evidence.tar.gz`, `experiments/independent-endpoint/summary.json` | 200-page gallery, original 1,033 query texts; pinned processor, native raw forward. Never substitute for the historical 3,420-page table. |

## Historical corrections that must travel with results

- The duplicate-contaminated pooled run is archived and never used in a headline.
- Shift empty queries were removed after the original run. Its pages remain as
  distractors. Of the 1,200 bridge-fitting queries, 127 were empty Shift texts;
  the historical bridges were not refitted.
- The original unstable reference argsort was corrected to stable tie handling.
  The original result JSON remains unchanged; `analyze.load()` recomputes metrics.
- Statements that all four pooled targets have a BF16 row are incorrect. The
  later 200-page check does not fill unmeasured 3,420-page entries.
- MMDocIR timing includes a previously measured raw-page reference. Preserve
  this provenance when quoting its complete-transition comparison.

No training, new benchmark, planner or cost-model experiment is added here.

The new endpoint has completed: each of four targets has 200 equal pages,
95,869,056 equal elements, zero max error, and 1,033 equal ordered top-10 lists.
[Saved pooled ordered-rank verification](ordered-ranking-check.json) separately
confirms ColQwen2 on the 3,420-page gallery without any GPU recomputation.

The 580 MMDocIR distractors now have a public-source row/hash manifest and
`experiments/prepare_mmdocir_distractors.py`; all image byte strings were compared
with the historical input and matched. No GPU benchmark was rerun for this check.

Source SHA-256 entries in `provenance.json` certify the imported snapshot, not
that the historical run captured that exact code hash. The pooled protocol
records post-run runner amendments; only the new endpoint has a pre-execution
code/weight/processor/data manifest.

Strict payload-bit verification is recorded in
`experiments/independent-endpoint/bitwise-verification.json`, with the hashes of
the externally retained tensor banks. Both public MMDocIR source reconstruction
and Energy logical-input reconstruction were executed on CPU; receipts are
in their preparation directories.

### Page-scoped processor refinement

`experiments/page-validity/README.md` maps the frozen 200-page CPU check and
16-page independent GPU boundary check to the new `reprforge/page_reuse.py`
mechanism. Raw records are in `evidence/page-validity-evidence.tar.gz`.
