# Section 4: comparable evidence and dataset-centered figures

This script organizes existing evidence; it runs no GPU experiment and changes
no historical protocol. Main comparisons use A100, BF16, and page batch 1.
Different galleries, processor budgets and reference implementations remain
explicitly separate. No 4090-to-A100 scaling or old raw baseline substitution.

```bash
git lfs install --local
git lfs pull
python experiments/verify_artifacts.py --extract-to /new/evidence/path
python -m pip install -r experiments/paper-section4/requirements.txt
python experiments/paper-section4/build.py \
  --evidence /new/evidence/path --output /path/to/section4-materials
```

Outputs: five English PDF/PNG figures, LaTeX table fragments, CSV/JSON data and
an input/script hash manifest. The author-facing Chinese LaTeX narrative lives
separately in the workspace's `paper/section4-experiments-20260911`, so rebuilding
plots does not rewrite manuscript prose. Output paths should be dedicated to
these generated files; regeneration replaces generated figures/tables/data.

## Main evidence

1. `01-quality-vs-fidelity`: named ColQwen2 v0.1 → v1.0 release; five public
   query datasets on a common 3,420-page gallery. Every top-k metric is rebuilt
   from ordered rank files and corrected query masks. The historical target
   reference shares its prefix; this is not an independent raw endpoint.
2. `02-matched-encoding-time`: ColQwen2.5 v0.1 → v0.2 native raw and persisted
   replay, same 200 pages, 12.8M processor. Ratios of sums, not averaged page
   speedups. Warm state read included; index build and initial capture excluded.
3. `03-mechanism-conditions`: fixed-page interventions on embeddings, vision,
   merger and positions. Per-page errors versus independent target forwards.
   These are causal diagnostics, not additional released transitions.

## Supplementary evidence

4. `04-processor-boundary`: identical input counts by dataset for a budget
   change. The page fraction is not a time-saving fraction.
5. `05-compression-tradeoff`: codecs within one ColQwen2 / 602k / 3,420-page
   protocol; all compressed routes are approximate.

Additional targets are listed by model name in `independent-targets.tex`, under
one pinned base/processor. They do not represent distinct model families.

`selection.json` records inclusions and exclusions. `data/manifest.json` records
the hashes of all used raw files, the builder, software versions, and micro/macro
summary values. Descriptive fixed-set results do not have population confidence
intervals. Per-page timing SD describes heterogeneity, not repeated-run error.

## Claim-focused redesign

`redesign.py --data /path/to/previous/data --output /new/figures` draws a result
retention histogram, ordered-ranking agreement, and matched aggregate encoding
time. Existing and independent-native reference scopes remain explicitly named.
After new experiments complete, `--full /completed/full-collection` and
`--mechanism /completed/mechanism-200` consume their actual results. No pending
or predicted measurements are drawn. See `../full-collection-endpoint/`.

`state_choice.py --pages /path/to/pages.jsonl --output /new/figures` pairs a
state/reconstruction schematic with the two prespecified control and embedding
conditions. It refuses missing, repeated or incomplete page sequences. The
current author preview uses the completed 200-page conditions while the full
five-condition run remains pending; it does not extrapolate missing outcomes.
The cost panel also shows individual pages and fixed-bin medians by document
vector count, without fitting a complexity law or inventing confidence intervals.
