# Batch-size control and in-domain residual MLP

## Batch size 1 vs batched control (same A100, same holdouts; DocVQA batch 2 because batch 4 exceeds 80 GB)

| Corpus | batch | preprocess s/page | prefix s/page | suffix s/page | leverage | replay PCA-256 s/page | saving | TA@10 bf16 | TA@10 pca256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arxivqa | 1 | 0.132 | 0.926 | 0.193 | 0.845 | 0.193 | 84.6% | 1.000 | 0.891 |
| arxivqa | 4 | 0.135 | 2.120 | 0.314 | 0.878 | 0.314 | 87.8% | 1.000 | 0.887 |
| docvqa | 1 | 0.233 | 1.812 | 0.291 | 0.876 | 0.291 | 87.5% | 1.000 | 0.853 |
| docvqa | 2 | 0.254 | 2.859 | 0.407 | 0.884 | 0.407 | 88.4% | 1.000 | 0.845 |
| flickr | 1 | 0.010 | 0.054 | 0.056 | 0.535 | 0.055 | 54.4% | 1.000 | 0.899 |
| flickr | 4 | 0.011 | 0.053 | 0.020 | 0.765 | 0.020 | 76.5% | 1.000 | 0.904 |

Batching amortises per-forward overhead. The question is whether it removes the
cut's advantage: if leverage at batch 4 stays close to batch 1 on document pages,
the paper's batch-1 timings do not flatter the cut. Small-image (Flickr)
leverage is expected to move most.

## In-domain Drift-Adapter residual MLP

| Corpus | fit queries | epochs | val. token cosine before → after | stale TA@10 | MLP in-domain TA@10 | MLP ΔnDCG@5 [95%] | stale ΔnDCG@5 |
|---|---:|---:|---|---:|---:|---|---:|
| arxivqa | 350 | 50 | 0.703 → 0.807 | 0.711 | 0.705 | -0.0137 [-0.0349,+0.0091] | -0.0236 |
| docvqa | 350 | 33 | 0.718 → 0.821 | 0.629 | 0.597 | -0.0032 [-0.0266,+0.0210] | +0.0110 |
| flickr | 700 | 50 | 0.789 → 0.875 | 0.817 | 0.783 | -0.0229 [-0.0368,-0.0099] | -0.0128 |

Same recipe as the paper's Energy-fitted bridge (hidden 256, dropout 0.1, AdamW
3e-4, token batches of 256, early stopping on a 20% validation split), fitted on
the non-holdout 70% queries of each corpus. Token cosine rises as before; TA@10
does not move away from stale serving.
