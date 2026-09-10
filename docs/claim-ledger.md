# Claim ledger

This ledger governs public descriptions. Historical reports and variable names
are retained for provenance and may use stronger terminology. Paths below are
relative to `experiments/`; raw numerical evidence is in the LFS archive.

| Claim | Strongest evidence | Scope / condition | Safe wording | Unsafe wording |
|---|---|---|---|---|
| Upgrade locality | `2026-09-04-public-release-stage-census-v1/raw-output/census.json` | 19 inspected retrieval adapter releases; tensor headers, not a weight-level proof of the full pipeline | No inspected release contained vision-tower or merger LoRA tensors | All adapters change only the suffix |
| Upstream validity | ColQwen2 CPU/GPU release protocol; processor certificate; dependency trace | Collection, processor/tokenization, vision, merger, text embeddings and numerical execution must be compatible with the cut | Reuse is conditional on unchanged or certified-equivalent upstream dependencies | Adapter-only means reusable; equal base names certify all dependencies |
| Clean official transition | `2026-08-22-colqwen2-release-transition-v1/{cpu-result,gpu-result}.json` | v0.1 → v1.0, same recorded base and shipped 602,112-pixel processor | A real official transition admits replay without changing the recorded shipped processor budget | Most real upgrades are like this |
| Independent representation equality | ColQwen2 release GPU result | 128 Energy pages, batch 4, BF16; target reload and persisted source IR | All tested target document elements equal the independent raw target endpoint | All transitions/pages are bitwise exact; physical index equality |
| Pooled historical BF16 rows | `2026-09-07-pooled-gallery-upgrade-matrix-v1`; `2026-09-09-colqwen2-pooled-upgrade-v1` | Same prefix tensor feeds target and BF16 suffix paths; cross-version canary on one page | Shared-prefix suffix consistency checks, with ranking metrics as recorded | Independent raw-versus-stored-state proof; four pooled targets ran BF16 |
| New cross-vendor endpoint | `independent-endpoint/summary.json`, raw pages/rankings in LFS | Four targets; separate source/target processes; native raw forward; 200 pages; pinned common 3B base and processor | Each target: 200/200 pages, 95,869,056/95,869,056 elements equal, zero max error; ordered top-10 equal for 1,033/1,033 queries over this 200-page gallery | Vendor-default processor behavior; general equality outside tested inputs |
| Target ranking fidelity | Pooled `rankings.json`, `per_query.json`, `extras.py` | TA@k is set overlap. RBO and Kendall are additional metrics, not ordered equality. New endpoint records ordered top-10 explicitly; saved pooled rankings checked in `docs/ordered-ranking-check.json` | ColQwen2 BF16 pooled route has 1,033/1,033 ordered top-10 matches to its shared-prefix target reference; independent 128-page representation evidence is separate | TA@10=1 proves ordered top-10 or representation equality |
| Physical index equality | None | No raw/replay serialized-index byte comparison | Not established | Reproduces the target index bit for bit |
| Encoding/replay savings | Pooled stage clocks | A100, batch 1, BF16; 12,845,056 versus 602,112 pixels; no state IO or index build | Suffix replay uses about 9.3% / 20% of raw page encoding time under these conditions | End-to-end 10× speedup; unconditional 90% saving |
| Complete transition savings | `2026-08-22-mmdocir-version-transition-v1/result.json` | Approximate PCA-256, warm A100, raw baseline frozen from previous run, 20,395 pages | Measured replay transition 1,351 s versus a 5,078 s reference assembled with the prior raw page baseline | Exact route; same-run paired raw measurement |
| RTX 4090 build savings | `2026-08-24-paper4090-main-table-repeats-v1/raw-output/` | Approximate PCA-256/Light10; per-collection batches; excludes loading, queries and calibration | State timing inclusions and approximation with the 77.8–85.2% range | Exact replay efficiency or a GPU-independent saving |
| Processor change | Census + `2026-08-22-processor-equivalence-certificate-v1/result.json` | v0.2 changes max_pixels; a different-file processor pair has equal observed outputs on 2,225 pages | Processor is versioned; changed files require conservative invalidation or scoped output certification | Every file difference implies changed outputs; equal max_pixels certifies a whole processor |
| Base-model change | Dependency trace; census | Static dependency evidence; no deliberate incompatible-base GPU feed | Invalidate when a changed base affects the cut's upstream dependencies or equivalence is unproven | Experimentally demonstrated wrong replay after arbitrary base change |
| Cross-vendor 3B locality | Census analysis: same LFS digest of shard containing vision + merger, two sampled vision tensor hashes | Only these 3B releases; metadata digest is not a fresh full local weight comparison | Shared vision/merger weight shard in the inspected 3B ecosystem | All vendors share byte-identical upstream computation; Nomic 7B is compatible |
| Metric-AI shipped contract | Recorded max_pixels equals v0.1 | Other processor fields not certified; endpoint uses fixed base processor | Same recorded pixel budget, evaluated under a pinned common processor | No pinning required; full factory processor equivalence proven |
| Compressed replay | Codec frontier, pooled PCA/INT8 routes | Lossy; fidelity depends on codec, collection and target | Approximate replay with reported ranking/quality metrics | Exact, lossless, or bitwise replay |

The historical census analyzer's `deepest_legal_cut` field checks same base and
vision-LoRA absence, but not its recorded processor delta. It is a structural
candidate label, not a complete reuse certificate. Consult full contracts.

The current paper phrases flagged by the preceding audit remain author edits:
unqualified cross-vendor byte identity, four-target independent exactness,
unrestricted “all adapters”, unconditional “about 90%”, and “vision tower output”
for a post-merger cut. No manuscript prose is rewritten by this repository update.

## Completed independent endpoint (2026-09-10)

The four ColQwen2.5 target adapters are vidore v0.2, Metric-AI 3B, T-Systems 3B
and ColNomic 3B, each applied to the same pinned Qwen base/processor contract.
All four passed the independent endpoint check. This upgrades cross-vendor
representation evidence but does not fill historical unmeasured full-pool rows.
Replay/raw page-time ratios including warm state reads were 12.90%, 13.26%,
13.23%, and 12.95%. They are descriptive single-pass paired timings on a shared
A100, with model loading, scoring, validation and serialization excluded.

For the supplied Abstract/Introduction, “all tested upgrades” must identify the
reuse-eligible BF16 routes and their test scope. The 9%–24% range comes from
stage/page-encoding measurements, not full index rebuild measurements. The
ColQwen2 ordered-top-10 statement now has direct saved-ranking verification;
its reference still uses the shared prefix, with independent representation
evidence supplied by the separate Energy check. No prose is rewritten here.

Post-run CPU verification of the persisted banks also confirmed bitwise BF16
tensor payload equality (including signed zero), 200/200 pages per target. See
`experiments/independent-endpoint/bitwise-verification.json`; bank hashes are
recorded there. This still does not establish physical index-file equality.
