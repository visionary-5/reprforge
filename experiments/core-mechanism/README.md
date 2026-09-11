# Core replay mechanism: controlled evidence

Purpose: test the conditions of the existing replay mechanism, not add another
model or feature. Read [`../../docs/replay-contract.md`](../../docs/replay-contract.md)
for the state-sufficiency argument and the concrete dependency boundary.

The frozen protocol uses the first eight pages of the existing independent
endpoint sample, covering six collections. Source v0.1 states were loaded from
disk, hash-checked, and replayed through an independently loaded v0.2 target.
The reference calls native full encoding from freshly processed raw images.
All conditions use the same pages, base, 12.8M processor, BF16 and batch 1.

Except the release control, conditions are deliberately synthetic interventions
or implementation ablations. They establish causal behavior in this integration,
not frequency of these changes in actual model releases.

| Condition | Comparison to independent native target | Bitwise-equal pages |
|---|---|---:|
| Real adapter transition under fixed processor | Visual state + target reconstruction | 8/8 |
| Target text embedding changed | Visual state + target text reconstruction | 8/8 |
| Same embedding change | Retain old fused text embeddings | 0/8 |
| Target vision patch weights changed | Force old visual state replay | 0/8 |
| Same vision change | Execute raw fallback independently | 8/8 |
| Target merger bias changed | Force old visual state replay | 0/8 |
| Same merger change | Execute raw fallback independently | 8/8 |
| Original target, position reconstruction ablated | Zero replay positions | 0/8 |
| Original target, positions intact | Normal replay | 8/8 |

All comparisons preserve output shape and yield finite outputs. Native encoding
calls the visual module once per page; replay calls it zero times, verified by
a forward hook. These call counts establish skipped computation, not elapsed-time
savings. Existing stage timing remains the efficiency evidence.

Maximum absolute element error ranges across the eight pages:

- stale fused text after embedding change: 0.423828125–0.4873046875;
- forced replay after vision change: 0.3828125–0.4814453125;
- forced replay after merger change: 0.35546875–0.4833984375;
- zero-position ablation: 0.392578125–0.521484375.

These are diagnostic error magnitudes for fixed interventions, not a robustness
curve. No perturbation sweep, ranking metric or statistical population inference
is claimed. Target parameter snapshots are restored between conditions and their
before/after hashes are recorded. The intervention magnitudes were frozen before
execution. The stale fused-text baseline reconstructs original text embeddings
from the shared base; it is not a historical independently saved fused checkpoint.

## What this resolves

1. State choice changes dependency requirements: the existing visual-only state
   can absorb target embedding-table changes by rebuilding text, whereas fused
   saved inputs contain stale text values.
2. Matching shapes and release labels are insufficient: upstream weight changes
   cause same-shape representation errors.
3. Dependency validity alone is insufficient if the resume implementation omits
   required context. Position reconstruction is independently necessary on these
   tested pages.
4. A dependency-based fallback restores the native target reference under the
   controlled invalidating updates.

This does not prove that arbitrary declared dependencies are complete. The
library's dependency checker consumes declarations; model integration must
establish them. See the limitations in the replay contract note.

## Reproduce

Use the pinned `experiments/requirements-gpu.txt` environment and a completed
`independent-endpoint` output directory with source states and its manifest.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/core-mechanism/run.py \
  --prior /path/to/independent-endpoint-output --output /new/output/path
```

The command refuses existing output directories. `result.json` is copied from
the raw run. `evidence/core-mechanism-evidence.tar.gz` contains the raw per-page
records, summary, log, and executed code/protocol. Tensor banks remain outside
Git and are identified by SHA-256 in the summary.

## Expanded validation (v2, results pending)

`run_v2.py --pages 200` uses every page in the same frozen prior sample and keeps all five
interventions unchanged. This successor was launched on 2026-09-11 to check
coverage beyond the original diagnostic eight pages. It writes its effective
protocol before model execution and refuses an existing output directory.
Do not interpret the v1 table above as the result of this expanded run.
