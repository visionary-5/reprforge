# Codec frontier and build-cost anatomy (NVIDIA A100-SXM4-80GB)

Holdouts, splits, adapters and processor are identical to the paper's Table 2 run
(`sigir-version-evolution-matrix-v1`); image batch size 1.

## Frontier: fidelity vs retained bytes

| Route | arxivqa TA@10 / ΔnDCG@5 [95%] / IR÷terminal(fp32) | docvqa TA@10 / ΔnDCG@5 [95%] / IR÷terminal(fp32) | flickr TA@10 / ΔnDCG@5 [95%] / IR÷terminal(fp32) |
|---|---|---|---|
| stale (legacy index) | 0.709 / -0.0236 [-0.0516,+0.0039] | 0.625 / +0.0129 [-0.0159,+0.0427] | 0.816 / -0.0114 [-0.0246,-0.0001] |
| Procrustes, in-domain | 0.727 / -0.0204 [-0.0440,+0.0035] | 0.637 / +0.0044 [-0.0205,+0.0306] | 0.825 / -0.0081 [-0.0199,+0.0031] |
| affine ridge, in-domain | 0.704 / -0.0093 [-0.0303,+0.0135] | 0.606 / -0.0062 [-0.0286,+0.0159] | 0.789 / -0.0138 [-0.0259,-0.0027] |
| exact BF16 | 1.000 / +0.0000 [+0.0000,+0.0000] / 7.97× | 1.000 / +0.0000 [+0.0000,+0.0000] / 7.98× | 1.000 / +0.0000 [+0.0000,+0.0000] / 7.63× |
| INT8/token | 0.973 / -0.0013 [-0.0035,+0.0000] / 3.99× | 0.955 / +0.0006 [+0.0000,+0.0015] / 3.99× | 0.984 / -0.0007 [-0.0019,+0.0000] / 3.82× |
| INT4/group64 | 0.947 / -0.0008 [-0.0081,+0.0066] / 2.12× | 0.926 / +0.0080 [-0.0076,+0.0274] / 2.12× | 0.959 / +0.0013 [-0.0045,+0.0074] / 2.03× |
| PCA-512 INT8 | 0.937 / -0.0021 [-0.0097,+0.0051] / 1.00× | 0.912 / +0.0159 [+0.0019,+0.0317] / 1.00× | 0.938 / +0.0008 [-0.0054,+0.0071] / 0.96× |
| PCA-256 BF16 | 0.886 / -0.0012 [-0.0132,+0.0119] / 1.00× | 0.853 / -0.0024 [-0.0181,+0.0141] / 1.00× | 0.898 / +0.0056 [-0.0045,+0.0159] / 0.95× |
| PCA-256 INT8 (paper) | 0.891 / -0.0025 [-0.0150,+0.0109] / 0.50× | 0.853 / -0.0031 [-0.0177,+0.0119] / 0.50× | 0.899 / +0.0030 [-0.0090,+0.0145] / 0.48× |
| PCA-128 INT8 | 0.772 / -0.0178 [-0.0363,-0.0023] / 0.25× | 0.753 / +0.0016 [-0.0145,+0.0187] / 0.25× | 0.801 / -0.0003 [-0.0125,+0.0116] / 0.24× |
| PCA-64 INT8 | 0.490 / -0.2523 [-0.3106,-0.1966] / 0.13× | 0.419 / -0.2672 [-0.3362,-0.1992] / 0.13× | 0.643 / -0.0701 [-0.0982,-0.0433] / 0.12× |

Reading guide: TA@10 is Top-10 agreement with the exact v0.2 index. The exact BF16
cut is the correctness check for the replay path (predicted 1.000). Every row
below it trades bytes for fidelity; the paper's PCA-256/INT8 is one point on
this curve, not the method.

## Anatomy: where raw build time goes

| Corpus | visual tok/page | preprocess s/page | prefix s/page | suffix s/page | leverage (pre+prefix)/total | replay PCA-256 s/page | measured saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| arxivqa | 2826 | 0.132 | 0.926 | 0.193 | 0.845 | 0.193 | 84.6% |
| docvqa | 4354 | 0.233 | 1.812 | 0.291 | 0.876 | 0.291 | 87.5% |
| flickr | 226 | 0.010 | 0.054 | 0.056 | 0.535 | 0.055 | 54.4% |

Leverage is the fraction of raw encode time spent before the language-model
suffix; no post-vision replay can save more than this. The measured saving is
replay (decode + suffix) against preprocess + prefix + suffix, excluding index
construction, which both routes share.

Flickr caveat: the paper's RTX 4090 table reports a 77.8% Flickr saving, but on
this A100 the same route saves 54.4% because a 226-token page is launch-latency
bound: the language-model suffix costs 0.056 s regardless of tokens, the same as
the whole visual prefix. At image batch 4 the suffix amortises to 0.020 s/page
and the saving is 76.5%, matching the 4090 (see extra-report.md). Small-image
savings depend on suffix batching, and the paper reports both regimes.

## Cross-backbone leverage (40 holdout pages per corpus, batch size 1)

| Backbone | vision params / total | corpus | doc vectors/page | preprocess s | vision s | forward s | vision share of forward | leverage |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| colpali | 14% | arxivqa | 1031 | 0.025 | 0.012 | 0.052 | 0.23 | 0.48 |
| colpali | 14% | docvqa | 1031 | 0.041 | 0.012 | 0.052 | 0.23 | 0.57 |
| colpali | 14% | flickr | 1031 | 0.008 | 0.012 | 0.051 | 0.23 | 0.33 |
| colqwen2.5 | 9% | arxivqa | 2800 | 0.132 | 0.902 | 1.137 | 0.79 | 0.81 |
| colqwen2.5 | 9% | docvqa | 4723 | 0.250 | 1.939 | 2.330 | 0.83 | 0.85 |
| colqwen2.5 | 9% | flickr | 239 | 0.012 | 0.055 | 0.095 | 0.58 | 0.63 |
| colqwen2.5 | 18% | arxivqa | 2800 | 0.134 | 0.904 | 1.043 | 0.87 | 0.88 |
| colqwen2.5 | 18% | docvqa | 4723 | 0.252 | 1.943 | 2.169 | 0.90 | 0.91 |
| colqwen2.5 | 18% | flickr | 239 | 0.011 | 0.056 | 0.109 | 0.52 | 0.56 |
| colsmol | 18% | arxivqa | 854 | 0.236 | 0.030 | 0.074 | 0.40 | 0.86 |
| colsmol | 18% | docvqa | 1086 | 0.317 | 0.038 | 0.081 | 0.46 | 0.89 |
| colsmol | 18% | flickr | 914 | 0.225 | 0.032 | 0.076 | 0.42 | 0.85 |

A backbone whose vision tower is a small fraction of the forward pass (ColPali:
SigLIP-400M in front of Gemma-2B) leaves little for a post-vision cut to save;
this is the architectural reason the paper's ColPali route saved under 20% and
failed admission. ColSmol's leverage is high only because Idefics3 image
splitting is CPU-heavy; the cut saves preprocessing rather than GPU time.

## Explained variance of the collection basis

- arxivqa: r=32: 0.563, r=64: 0.690, r=128: 0.810, r=256: 0.911, r=512: 0.973
- docvqa: r=32: 0.603, r=64: 0.712, r=128: 0.825, r=256: 0.925, r=512: 0.980
- flickr: r=32: 0.566, r=64: 0.714, r=128: 0.838, r=256: 0.924, r=512: 0.974

## Claim boundary

One backbone, one release pair, one GPU. In-domain bridges are fitted on the
non-holdout queries of the same corpus, so they have an advantage the paper's
Energy-fitted bridges did not; if they still do not approach codec TA@10, the
compatibility-vs-target-semantics distinction does not depend on the fitting domain.
