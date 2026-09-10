# Statistical Appendix

## Protocol and unit of analysis

- Hardware: one NVIDIA A100-SXM4-80GB, GPU 0.
- Workload: 128 Energy pages with frozen processor signature `[1, 64, 46]`.
- Language batch size: 8.
- Repeats: 3 complete builds per method, with rotated method order.
- Unit: one complete 128-page run.
- Metric: wall-clock seconds, lower is better; source preparation/load and model
  execution are included.

## Exact timing records

| Repeat | Raw Full | IR Full | Raw L12 | IR L12 |
|---:|---:|---:|---:|---:|
| 0 | 38.947 | 6.198 | 38.477 | 4.379 |
| 1 | 39.560 | 6.209 | 38.030 | 4.375 |
| 2 | 40.190 | 6.301 | 37.286 | 4.381 |

## Descriptive and uncertainty statistics

| Contrast | Baseline mean ± SD | Reuse mean ± SD | Paired saving mean ± SD | Mean relative saving | 95% t interval for paired seconds |
|---|---:|---:|---:|---:|---:|
| Full: raw vs IR | 39.566 ± 0.621 | 6.236 ± 0.057 | 33.330 ± 0.570 | 84.24% | [31.914, 34.745] |
| L12: raw vs IR | 37.931 ± 0.601 | 4.378 ± 0.003 | 33.553 ± 0.603 | 88.46% | [32.056, 35.050] |

The t intervals use `t(0.975, df=2)=4.303`. With only three repeats, normality
cannot be meaningfully established; these intervals illustrate run-level
uncertainty and should not be interpreted as cross-machine or cross-workload
population intervals.

## Inferential checks and effect sizes

- All three paired differences are positive for Full and L12.
- An exact two-sided sign test with 3/3 wins gives `p=0.25` per contrast. With a
  Holm correction over the two preregistered reuse contrasts, adjusted `p=0.50`
  for each. Therefore this small screen does not support conventional
  significance wording.
- Matched-pairs rank-biserial direction effect is `r=1.0` for both contrasts.
- Standardized paired effects are `d_z=58.48` (Full) and `55.67` (L12), but these
  values are dominated by extremely small timing variance and are unstable at
  `n=3`; raw seconds and relative savings are more interpretable.

No observation was removed as an outlier. No post-hoc contrast was used to
select the main finding.

## Equivalence

| Comparison | Pages | Tensor equality | Maximum absolute error |
|---|---:|---:|---:|
| Raw Full vs IR Full | 128 | true | 0.0 |
| Raw L12 vs IR L12 | 128 | true | 0.0 |

This establishes source-path equivalence for the frozen code and checkpoint,
not model-family equivalence.

## Stage decomposition

| Method | Source/preparation mean | Language + projection/output mean | Total mean |
|---|---:|---:|---:|
| Raw Full | 33.609 s | 5.955 s | 39.566 s |
| IR Full | 0.286 s | 5.947 s | 6.236 s |
| Raw L12 | 33.844 s | 4.085 s | 37.931 s |
| IR L12 | 0.298 s | 4.079 s | 4.378 s |

IR reuse removes the observed 33.3–33.6 s source/vision stage while leaving
suffix time unchanged. This stage localization, together with exact output
equality, supports the dependency-reuse mechanism more directly than the total
speedup alone.

## Storage and amortization

| Artifact | Total | Per page | Ratio to Full index |
|---|---:|---:|---:|
| Compressed source images | 43.00 MB | 0.336 MB | 0.88× |
| Reusable BF16 vision IR | 397.64 MB | 3.107 MB | 8.12× |
| Full float32 terminal index | 48.96 MB | 0.382 MB | 1.00× |
| L12 float32 terminal index | 24.84 MB | 0.194 MB | 0.51× |

The measured serialization write time was 0.453 s. Dividing it by the observed
mean per-recompile savings gives 0.014 Full updates or 0.013 L12 updates. A more
conservative observed initial-build overhead compares cache creation plus the
IR suffix against Raw Full: approximately `(34.321 + 5.947) - 39.566 = 0.702 s`,
which is recovered after about 0.021 Full recompiles. These wall-time ratios omit
the monetary and capacity cost of retaining 397.64 MB, so they are not a complete
economic break-even analysis.

## Statistical blockers

- `n=3` blocks a persuasive significance claim despite large observed effects.
- Repeats on one host do not measure hardware variability.
- A single shape reduces scheduler noise but does not establish corpus-level
  heterogeneity performance.
- No cold-storage measurement and no storage-cost model are available.
