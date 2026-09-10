# Frozen predictions (colqwen2-pooled-upgrade-v1)

| id | holds | evidence |
|---|---|---|
| C1_exact | True | exact TA@10 1.000; bitwise-identical pages 40/40; max abs error 0.00e+00 |
| C2_stale_breaks | True | stale TA@10 0.496, top-1 flip 0.304, dnDCG -0.055 |
| C3_operating_point | True | PCA-512 TA 0.869 dnDCG -0.001; PCA-256 TA 0.748 dnDCG -0.006 |
| C4_leverage | False | leverage 0.758; tokens/page 707 |
| C5_hot_refresh | True | matched hot-refresh TA@10 0.388 at fraction 0.242 |
