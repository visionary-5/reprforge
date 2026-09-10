# Pooled gallery: 3420 pages, 1033 queries, max_pixels 602112

Visual tokens/page 707; cut leverage colqwen2-v1.0 0.758

## colqwen2-v1.0 (same-vendor official release, same base and processor as shipped); matched hot-refresh fraction 0.242

| route | nDCG@5 | dnDCG@5 [95%] | R@1 | top-1 flip | TA@5 | TA@10 | payload B/page | GPU s/page |
|---|---|---|---|---|---|---|---|---|
| full re-encode (reference) | 0.800 | — | 0.719 | 0.000 | 1.000 | 1.000 | — | 0.163 |
| stale index (no-op) | 0.745 | -0.055 [-0.070,-0.041] | 0.648 | 0.304 | 0.527 | 0.496 | — | 0.000 |
| Procrustes bridge | 0.764 | -0.036 [-0.050,-0.022] | 0.683 | 0.281 | 0.552 | 0.535 | — | 0.000 |
| affine ridge bridge | 0.744 | -0.056 [-0.072,-0.040] | 0.665 | 0.302 | 0.519 | 0.501 | — | 0.000 |
| hot-refresh, matched GPU | 0.570 | -0.230 [-0.255,-0.206] | 0.475 | 0.478 | 0.401 | 0.388 | — | 0.039 |
| hot-refresh 25% | 0.569 | -0.231 [-0.256,-0.207] | 0.473 | 0.479 | 0.402 | 0.390 | — | 0.041 |
| hot-refresh 50% | 0.611 | -0.189 [-0.213,-0.166] | 0.519 | 0.378 | 0.574 | 0.568 | — | 0.081 |
| exact cut (BF16) | 0.800 | +0.000 [+0.000,+0.000] | 0.719 | 0.000 | 1.000 | 1.000 | 2.17 MB | 0.039 |
| cut INT8/token | 0.800 | +0.000 [-0.001,+0.002] | 0.722 | 0.018 | 0.966 | 0.969 | 1.09 MB | 0.039 |
| cut PCA-512 INT8 | 0.799 | -0.001 [-0.006,+0.003] | 0.716 | 0.067 | 0.872 | 0.869 | 0.36 MB | 0.039 |
| cut PCA-256 INT8 | 0.794 | -0.006 [-0.015,+0.002] | 0.712 | 0.123 | 0.749 | 0.748 | 0.18 MB | 0.039 |
| cut PCA-128 INT8 | 0.715 | -0.086 [-0.103,-0.069] | 0.614 | 0.297 | 0.538 | 0.517 | 0.09 MB | 0.039 |
| full re-encode + token pooling x3 | 0.793 | -0.007 [-0.014,+0.000] | 0.710 | 0.120 | 0.823 | 0.820 | — | 0.000 |
| cut PCA-256 + token pooling x3 | 0.784 | -0.016 [-0.026,-0.006] | 0.697 | 0.148 | 0.743 | 0.731 | — | 0.000 |

## Storage denominators (bytes per page)

| object | bytes/page |
|---|---|
| terminal float32 | 0.368 MB |
| terminal bf16 | 0.184 MB |
| terminal pooled3_bf16 | 0.061 MB |
| terminal light10_sq8 | 0.009 MB |
| cut pca128_int8 | 0.092 MB |
| cut pca256_int8 | 0.183 MB |
| cut pca512_int8 | 0.364 MB |
| cut int8_token | 1.088 MB |
| cut bf16 | 2.173 MB |
