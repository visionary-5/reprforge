# Frozen predictions

| id | holds | evidence |
|---|---|---|
| Q1_spread | True | v0.2 target nDCG@5 on ArxivQA queries = 0.897 |
| Q2_cross_vendor_collapse | True | max per-collection stale nDCG@5: {'metric-ai-3b': 0.006666666666666667, 'tsystems-3b': 0.0}; v0.2 |stale-target| pooled = 0.023 |
| Q3_exact_cut | True | exact-cut TA@10: {'vidore-v0.2': 1.0, 'metric-ai-3b': 1.0} |
| Q4_top1_flip | True | v0.2 stale top-1 flip rate pooled = 0.199 |
| Q5_hot_refresh | True | hot-refresh@matched TA@10 {'vidore-v0.2': 0.49535333978702806, 'metric-ai-3b': 0.10280735721200387, 'tsystems-3b': 0.10135527589545015, 'colnomic-3b': 0.1565343659244918}; pca256 TA@10 {'vidore-v0.2': 0.8736689254598258, 'metric-ai-3b': 0.8849951597289448, 'tsystems-3b': 0.9013552758954502, 'colnomic-3b': 0.8636011616650533} |
| Q6_leverage_budget | True | leverage 12.8M px = 0.907; 602k px = 0.800 |
| Q7_fidelity_at_scale | True | pca256 - target nDCG@5: {'vidore-v0.2': -0.0024, 'metric-ai-3b': -0.0027, 'tsystems-3b': -0.0014, 'colnomic-3b': -0.0049} |
| Q8_composition | True | (TA pooled-cut vs pooled-target, pooled-target nDCG delta): {'vidore-v0.2': (0.86, -0.0003), 'metric-ai-3b': (0.874, -0.0067)} |
