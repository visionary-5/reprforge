# Reproduction guide

The core library runs on CPU; GPU experiments are optional. Historical runners
are archived snapshots (including documented amendments), not imported by the library. Keep outputs outside the Git
checkout. The [evidence index](../docs/evidence.md) identifies which experiment
supports each claim and which historical interpretations have been corrected.

## 1. Inspect the saved evidence without a GPU

Install Git LFS, then from the repository root:

```bash
git lfs pull
python experiments/verify_artifacts.py --extract-to /tmp/reprforge-paper-evidence
```

Use a new extraction directory: the verifier refuses overwrites. It checks code
and protocol hashes, archive digests, and every historical evidence member listed
in `docs/provenance.json`. The archive contains original JSON/per-query/ranking
outputs and historical analyses, including the excluded duplicate-contaminated
run. It contains no page images, trained models, IR states or embedding banks.

For example, regenerate the census offline (Python 3.12+ for its historical
analysis syntax):

```bash
cp experiments/2026-09-04-public-release-stage-census-v1/analyze.py \
  /tmp/reprforge-paper-evidence/2026-09-04-public-release-stage-census-v1/analyze.py
python /tmp/reprforge-paper-evidence/2026-09-04-public-release-stage-census-v1/analyze.py
```

Regenerate the corrected pooled tables, using the archive's original rankings,
per-query rows and query mask. This analysis needs NumPy and writes ignored
`analysis-output/` beside the script; it does not overwrite `tables/`:

```bash
python experiments/2026-09-07-pooled-gallery-upgrade-matrix-v1/analyze.py \
  /tmp/reprforge-paper-evidence/2026-09-07-pooled-gallery-upgrade-matrix-v1/raw-output/output-12845056 \
  /tmp/reprforge-paper-evidence/2026-09-07-pooled-gallery-upgrade-matrix-v1/raw-output/output-602112
```

Do not quote the census `deepest_legal_cut` column as a full validity certificate:
its analyzer omits the processor condition in that column. Do not quote pooled
raw aggregates before applying the documented Shift/tie corrections.

## 2. Minimal independent real-model check

Use an isolated Linux Python 3.11 environment with an NVIDIA GPU. The measured
setup used an A100 80 GB, BF16, image batch 1 and query batch 32. Install:

```bash
python -m pip install -e .
python -m pip install -r experiments/requirements-gpu.txt
```

Review [the frozen endpoint protocol](independent-endpoint/protocol.json) before
running. It explains the reviewer question, expected evidence, scope and failure
interpretation. This is a correctness experiment; shared-host timing is secondary.

Download exact model/data revisions and verify SHA-256. The LFS metadata bundle
preserves processor/tokenizer/config files and the small base projection fragments
used in the original runs. Full backbone and adapter weights come from their
public repositories, listed in [`sources.json`](independent-endpoint/sources.json).
Their original model licenses apply; the metadata bundle does not change them.
The datasets retain the terms of their ViDoRe source repositories.

```bash
python experiments/independent-endpoint/prepare_inputs.py --root /absolute/new/input-directory
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python experiments/independent-endpoint/run.py \
  --config /absolute/new/input-directory/endpoint-config.json \
  --output /absolute/new/endpoint-output
```

The input root and output directory must be new. No implicit overwrite or
model/data download occurs inside the GPU runner. To audit already prepared
inputs without network access:

```bash
python experiments/independent-endpoint/prepare_inputs.py \
  --root /absolute/new/input-directory --check-only
```

For an interrupted download, rerun preparation with `--resume`. Existing files
are verified and retained; a mismatching file is never silently overwritten.

The input preparation merges the two pinned TAT-DQA Parquet shards in order
using PyArrow 19.0.1; its reconstructed SHA-256 matches the historical merged
file. `HF_ENDPOINT` may point at your trusted Hub mirror if necessary; all bytes
are still checked against the same pinned digests.

The runner first hashes all local weights and processor files. It chooses 200
unique pages round-robin from six collections (34/34/33/33/33/33). The source
process writes only post-merger visual tokens and resume metadata, then exits.
Each target loads in a fresh process. Its raw route calls native model forward
from freshly decoded pixels; its replay route reads source states and recomputes
target text embeddings, positions and suffix. No raw-prefix tensor is reused.

Outputs include `manifest.json`, `inputs.json`, `source.json`, per-target
`*-pages.jsonl`, `*-rankings.json`, `*-result.json`, and final `result.json`.
Large `states/*.pt` and `*-banks.pt` remain in your external output directory.
The 1,033 queries are scored against **200 pages**, not the historical 3,420-page
pool. This tests ordered fidelity, not benchmark retrieval effectiveness.

## 3. Historical full experiments

Each dated directory contains its protocol, runner and original host launch
record. Use `--help` to map paths into your environment. The `run-*.sh` files
contain original author paths and GPU IDs; they are provenance, not portable
entry points to run unchanged. Shared helpers are included:

```bash
export PYTHONPATH="$PWD/experiments/support${PYTHONPATH:+:$PYTHONPATH}"
python experiments/2026-09-07-pooled-gallery-upgrade-matrix-v1/pooled_upgrade_matrix.py --help
python experiments/2026-09-04-codec-frontier-stage-anatomy-v1/run_codec_frontier.py --help
python experiments/2026-08-22-colqwen2-release-transition-v1/run_gpu.py --help
```

For pooled/frontier runners, pass `--matrix-code-root experiments/support` and
`--support-code-root experiments/support`. Copy `collections.json` and
`targets.json` to an external configuration directory and replace paths using
your prepared inputs. Keep the frozen protocol and hashes unchanged. The full
pooled run also uses 580 MMDocIR distractor pages. Their exact reconstruction
was checked against the historical input (580/580 image byte strings equal).
Download `MMDocIR_pages.parquet` from dataset
`MMDocIR/MMDocIR_Evaluation_Dataset` at revision
`bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4`, then run:

```bash
python experiments/prepare_mmdocir_distractors.py \
  --pages-parquet /absolute/path/MMDocIR_pages.parquet \
  --output /absolute/new/distractors.parquet
```

The script verifies the full source SHA-256 and every selected image hash using
`2026-09-07-pooled-gallery-upgrade-matrix-v1/mmdocir-distractors.json`.
Pass the resulting file as `--distractor-parquet`. Only the consumed image
column is written; the Parquet container need not match the old full-schema file.

The ColQwen2 pooled analyzer reuses the preceding ColQwen2.5 analysis loader;
both directories are shipped. The Vietnamese Energy runner is the same frozen
`run_colqwen_recompile.py` as the incremental experiment and uses its earlier
persisted cache. The RTX 4090 repeats use the three `*-v3-corpus-calibration`
runner directories and `infovqa-prospective-lifecycle-4090-v1` shipped here;
the repeats directory itself holds the protocol, host manifest and analyzer.
MMDocIR's publisher additionally needs FAISS and externally generated terminal
banks/IR described by its protocol. Its saved numerical result can be inspected
without those artifacts.

These historical full runs are not required to reproduce the new endpoint
check. The main pooled gallery is reconstructible from pinned public sources.
The [Energy preparation notes](data-preparation/README.md) recover the historical
logical inputs. A separate, explicit content-certified reproduction protocol
handles its Parquet serialization difference without editing the original
protocol. Generated historical MMDocIR IR/banks also
remain external. Recorded output hashes do not replace these input requirements
for verbatim reruns.

## Verification scope

CI runs CPU tests and examples, library lint and wheel packaging. Frozen GPU
scripts receive syntax/undefined-name lint without reformatting, preserving
source hashes. Their numerical behavior is checked by the explicit GPU endpoint
experiment, not by a synthetic CPU test pretending to execute the backbone.

The historical pooled runner snapshot already skips empty query strings, while
the original run encoded some Shift empty strings before the post-run correction.
Thus reanalysis of archived outputs exactly reproduces the published historical
rows; running the current snapshot is a corrected rerun and will not reproduce
the original bridge-fitting contamination. Do not claim its source hash was
captured before the historical run. The new independent endpoint does capture
executed source and protocol hashes before any GPU output.
