# Unified RTX 4090 main-table repeats

## Decisive result

| benchmark | pages / queries | Raw Light10 (s) | ReprForge Light10 (s) | saving | Light10 nDCG delta | IR / terminal |
|---|---:|---:|---:|---:|---:|---:|
| ArxivQA | 500 / 500 | 217.84 ± 1.84 | 32.18 ± 0.06 | 85.23% ± 0.10 pp | +0.01372 | 1.072× |
| DocVQA | 500 / 500 | 213.08 ± 2.75 | 34.70 ± 0.23 | 83.71% ± 0.31 pp | -0.00418 | 1.077× |
| Flickr | 1000 / 1000 | 111.00 ± 0.86 | 24.68 ± 0.01 | 77.76% ± 0.16 pp | +0.00811 | 1.060× |
| InfoVQA | 500 / 500 | 191.33 ± 0.82 | 34.79 ± 1.07 | 81.81% ± 0.56 pp | -0.00708 | 1.077× |

All twelve complete transitions passed dependency, exact-lowering, cardinality, and storage gates. Every paired run saved time. Across four retrieval surfaces, ReprForge reduced the measured adapter-transition target-build time by 77.76%--85.23% while the already frozen Light10 quality deltas all remained above the aggregate -0.01 gate.

The narrow within-host variation removes the earlier weakness that the main systems table mixed one-off measurements from different RTX 4090 instances. It does not create new query samples: the paired-query bootstrap intervals remain those of the original benchmark runs.

## Interpretation

The result supports the paper's lifecycle claim, not a generic encoder speedup claim. For a fixed processor and an adapter/projection-only target update, the corpus-calibrated reusable post-vision IR avoids recomputing the visual prefix and still materializes the target adapter's compact physical index. The benefit survives document-like pages, natural images, and two visual-QA surfaces under one implementation and one host.

## Statistical boundary

With n=3 technical repeats, no parametric significance test is reported. The analysis shows every raw/replay observation, mean, and sample SD. Retrieval uncertainty is query-level and comes from the previously frozen paired bootstrap: only ArxivQA has an interval whose lower bound clears -0.01; DocVQA, Flickr, and InfoVQA retain wider intervals despite passing the aggregate point gate.

## Reproducibility boundary

These repetitions estimate technical wall-time variability on one RTX 4090 host. They do not add independent retrieval samples, change the previously frozen paired-query intervals, or establish cross-hardware generality.
