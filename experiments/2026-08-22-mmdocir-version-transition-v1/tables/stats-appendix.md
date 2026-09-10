# Statistical appendix

- Unit for official quality inference: query (`n=1658`).
- Recall@5 contrast: +0.003126257; 20,000-draw paired query bootstrap 95% interval [-0.003558756, +0.009891687].
- nDCG@10 contrast: +0.001597008; descriptive here because the frozen primary inference gate was Recall@5.
- Global candidate probe: `n=20` deterministic, domain-stratified queries. Mean Full exact Top-10 candidate recall 0.985000. It is not an official benchmark estimate.
- Timing is one complete execution on a shared server. The comparison holds downstream serving/validation work constant and replaces only raw-page construction with IR replay. It is not a hardware-general confidence interval.
- All primary gates were frozen in `protocol.json` before full execution; no hyperparameter or threshold sweep followed observation of this result.
