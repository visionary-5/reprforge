# Changelog

## Unreleased — paper reproducibility

- Publish frozen GPU runners, protocols, input preparation, evidence indexes and
  Git LFS numerical evidence bundles.
- Add independent native-raw versus persisted-state checks for four ColQwen2.5
  target adapters, and verify ordered rankings from existing ColQwen2 outputs.
- Clarify representation/ranking/index equality and timing boundaries; retain
  all historical measurements and keep the core algorithm unchanged.

## 0.5.0 (2026-09-04)

Repository reorganised around the paper's claims.

- Flat package: `versions`, `dependencies`, `equivalence`, `planning`,
  `adapter`, `index`, `generation`. One module per concern.
- New: `cut_leverage` and `break_even_upgrades` in `planning`;
  `target_agreement` in `index`; `CutState` and the emit-cut / resume adapter
  contract in `adapter`; merger tensors classified as visual prefix and
  Idefics3 / ModernVBERT decoder paths recognised in `dependencies`.
- Index manifests now record the rebuild source (`raw` or a cut name) and the
  version tuple; the compile-plan fingerprint is gone (format 3).
- Removed the experimental in-flight lowering operator (topology-anchored
  assignment and coalescing), the Full/Compact lifecycle policy and query-time
  refinement. None of them is part of the paper's method or claims.

## 0.4.0

Last release with the experimental lowering operator.
