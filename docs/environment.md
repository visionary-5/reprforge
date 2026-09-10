# Environment and measurement boundaries

| Evidence | Recorded environment | Batch / processor / dtype | Timed scope |
|---|---|---|---|
| Pooled matrix and codec anatomy | A100-SXM4-80GB; Python 3.11, torch 2.5.0+cu124, transformers 4.53.2, PEFT 0.15.2, colpali-engine 0.3.12 | Images 1, queries 32; explicit max_pixels 12,845,056 or 602,112; BF16 | Synchronized preprocessing, prefix, suffix and codec stage clocks. No disk-state or index rebuild costs. |
| ColQwen2 release endpoint | A100-SXM4-80GB; see frozen GPU protocol | Images 4, 128 Energy pages, max_pixels 602,112 | Warm paired raw/replay page build, including IR read. Three technical repetitions; not independent datasets. |
| RTX 4090 repeats | See committed `host-manifest.json`; torch 2.5.1 | Images 4/1/8/1 by collection; 602,112; PCA-256/Light10 approximate routes | Compute, D2H, merge and terminal serialization; excludes loading, query encoding and one-time calibration. |
| MMDocIR transition | A100; protocol/provenance files | 20,395 pages, 768-token contract; approximate PCA-256 | Includes query encoding, SQ8 index build, validation, hashing and publication. Raw page baseline frozen from prior run. |
| New independent endpoint | Runtime `manifest.json` records exact installed versions and all input hashes | Images 1, queries 32, 12,845,056, BF16; fixed base processor | Raw includes image decode/preprocessing, transfer, native forward and D2H. Replay includes warm disk read, transfer, reconstruction, suffix and D2H. Excludes loading, validation, scoring and terminal serialization. Shared GPU: timings are descriptive, not a controlled efficiency headline. |

For the pooled/codec and new endpoint runs, cuDNN SDPA is disabled: the recorded
torch 2.5.0 setup produced nonfinite vision outputs with that backend. TF32 is
allowed. Changing kernels, batch shape, package versions or numerical precision
can change bitwise equality; rerun the independent endpoint check after such a
change. The exactness check records finite values and shape agreement as well
as `torch.equal`.

`requirements-gpu.txt` captures the historical A100 stack for an isolated Linux
Python 3.11 environment. It is separate from the CPU library's dependencies.
FAISS is required only by the historical MMDocIR index publication experiment.
Model/data acquisition and loading are never silently included in stage timing.
