# Statistical appendix

## Comparison contract

- Primary metric: complete 128-page target-version build wall time; lower is better.
- Paired unit: one raw/replay warm-worker repetition using the same fixed pages and checkpoints.
- Repetitions: 3 matched pairs in frozen alternating order.
- Hardware: one NVIDIA A100-SXM4-80GB.
- Multiple comparisons: none for the frozen primary contrast.
- Artifact provenance: local `gpu-result.json` SHA-256 `ad497dd90f5fea5229beeeedd5b2ecc0688af0b4c72f1a94c48a6705f5bb45c6`, identical to the remote checksum.

## Descriptive timing statistics

| Metric | Raw | Replay |
|---|---:|---:|
| Runs | 3 | 3 |
| Mean ± sample SD | 34.7254 ± 0.2190 s | 3.6909 ± 0.0110 s |
| Median | 34.6920 s | 3.6872 s |
| Min–max | 34.5250–34.9592 s | 3.6822–3.7033 s |

Matched raw-minus-replay differences are 31.2771, 31.0049, and 30.8218 s; mean ± SD is 31.0346 ± 0.2291 s. A t-based 95% interval is [30.4655, 31.6037] s, but it is reported only as a descriptive warm-run interval because normality cannot be meaningfully checked at n=3.

The operational effect sizes are:

- median time ratio: 9.4088×;
- median fractional saving: 89.3717%;
- minimum matched fractional saving: 89.2737%;
- paired wins: 3/3.

The exact two-sided sign-test p-value is 0.25. This correctly reflects that three technical repetitions are underpowered for a population-level inferential claim. The frozen physical gate instead requires at least 50% median saving and 3/3 paired wins; both conditions pass by a wide margin.

## Correctness and invalidation

Raw v1.0 versus replayed v1.0 compares 12,058,624 terminal elements: all are equal, maximum absolute error is 0, and mean token cosine is 1.0.

Source v0.1 versus target v1.0 compares the same number of elements: 12,031,863 differ (99.7781%), maximum absolute error is 0.4765625, and mean token cosine is 0.691388. This is a deterministic whole-slice comparison, not a statistical sample test.

## Storage accounting

| Artifact | Bytes | Relative to compressed images | Relative to float32 terminal |
|---|---:|---:|---:|
| Compressed images | 68,152,602 | 1.000× | 1.413× |
| Float32 target terminal | 48,234,496 | 0.708× | 1.000× |
| Exact post-vision IR | 297,150,528 | 4.360× | 6.161× |

IR construction, including source terminal generation, took 36.5606 s. That creation cost is deliberately excluded from a single successor rebuild time and must be amortized across lifecycle events; no break-even-frequency or storage-price inference is made here.

## Assumptions and blockers

- The three timings share a fixed model, dataset, process, GPU, and warm local-storage environment; they quantify repeatability, not cross-environment generalization.
- No normality or variance test is valid at n=3.
- Tensor equality implies identical document-side MaxSim inputs under a fixed query encoder and scorer, but this canary is not a full benchmark-quality run.
- The current exact IR fails the storage-admission objective. Compact-IR evidence comes from separate frozen experiments and must not be silently merged into this result.
- Cold cache, object storage, validation scans, concurrent load, and a second hardware class remain unmeasured for this release pair.
