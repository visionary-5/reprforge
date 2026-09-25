# Interpretation of the supplied records

Recorded evidence paths in this document refer to the accompanying supplementary
archive. Generated results are not stored in this Git repository.

## Released-pair audit

`evidence/all_rows.json` preserves 74 family-audit records. Two Qwen3 transition
summaries are supplied separately. Original verdict fields are historical
runner outputs and are not rewritten during packaging.

Sixteen family rows have verdict `ACCEPT` and also record the note
`native processor unusable (ValueError); base processor used`. On those rows,
the measured route used the family base processor after loading the release's
processor failed. The label alone therefore does not establish execution with
the target release's own processor. The full processor notes remain available
in every record and are surfaced by `scripts/summarize_records.py`.

One family record has the same repository identifier on both sides:
`vidore/colqwen2.5-v0.1`. The source input specification records revision
`d8bacd9564935c65f560e6aa841320e970bce02b`; the downloaded target revision record
is `4379ecfd4c9446836ad0ef4dd7390b1c748f68d8`. Equal repository names are not
sufficient to decide whether this is a changed-model transition. This row also
records the base-processor fallback above and identical source/target vectors
on its five-page diagnostic. It is retained as recorded, not relabelled as
proof that the embedding model changed.

## Lifecycle and generic validation

The 20,946-page lifecycle uses a common 602,112-pixel processing configuration
and fixed input contract, with physical ANN construction excluded. The ordinary
initial build is extrapolated from a measured 3,000-page run. Lifecycle totals
are computed before rounding. The fixed integration and historical dependency
contract are described in [implementation.md](implementation.md).

The separate 200-page generic-validation experiment uses warm retained-state
reads, compressed image bytes in RAM, three paired rounds and vector writes
with fsync. Its speedups are not substituted into the lifecycle measurements.

## Verification levels

CPU unit tests check implementation behavior on small synthetic inputs.
Record aggregation verifies calculations over supplied historical measurements.
Neither check re-executes the GPU experiments or independently verifies omitted
full vector banks. The archive does not claim one-command reproduction of every
appendix experiment; coverage is listed in [paper-map.md](paper-map.md).
