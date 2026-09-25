# Paper-to-code and evidence map

Recorded evidence paths in this document refer to the accompanying supplementary
archive. Generated results are not stored in this Git repository.

Section labels refer to the submitted paper. A runnable script is not by itself
a complete reproduction workflow: model snapshots, data manifests, environment
and loader integrations are also required. "Recorded" below means existing
measurements, not a fresh reproduction during packaging.

| Paper location | Implementation / entry | Included evidence and coverage |
|---|---|---|
| Method §3.1, §3.3 | `reprforge/tracing.py`, `reprforge/hooked.py` | Executed-path discovery and target-forward substitution; CPU mechanism tests in `tests/test_traced_replay.py`. |
| Method §3.2 | `hooked.dependency_digest`, `ResumeStub`; callers in `experiments/paper/item1` and `experiments/final_pass` | Loaded parameter/buffer checks and generic boundary-input checks. See implementation notes for fixed lifecycle differences. |
| Method §3.4 | `reprforge/planning.py`; fixed integration in `reprforge/integrations/colqwen.py` | Planning utilities are separate from the measured single-boundary lifecycle. |
| §4.2, architecture discovery | `experiments/paper/item3/`; Idefics3 path in `paper/item1/sweep_pairs.py` | `evidence/item3_table.json` and Qwen3 summaries. Qwen3 2B depends on unbundled producer/probe modules. |
| §4.3, released pairs | `experiments/paper/item1/sweep_pairs.py` | `evidence/all_rows.json` plus two Qwen3 summaries; processor fallback and same-repository revisions are explicitly described in `protocol-notes.md`. |
| §4.4, 200-page reconstruction | `experiments/reconstruction/run.py`, `verify_tensor_payloads.py` | Has frozen input preparation. Full historical vector banks are not distributed. |
| §4.4, 2,840-page reconstruction | `experiments/reconstruction/full_collection/` | Runners and protocols; requires regenerating target/reference vectors. |
| §4.5, lifecycle | `experiments/paper/e2e/` | Route summaries, corpus manifest, verification summaries and analysis in `evidence/lifecycle/`; final input layout is not automatically downloaded. |
| Appendix C, manual/automatic interface | `experiments/final_pass/manual_interface.py` | `evidence/final-pass/manual-qwen25-full/` and `manual-qwen3-full/`; model-specific environment and snapshots required. |
| Appendix C, 100-page Qwen3 2B | Original producer/probe integration | Not self-contained in this archive; eight-page traced checks are separate evidence, not a substitute. |
| Appendix C, replay-boundary diagnostics | Historical boundary probe | Final 200-page boundary runner and input closure are not supplied here. Planning examples do not reproduce this table. |
| Appendix D, controlled interventions | `experiments/ablation/run.py` | Runnable controlled Qwen2.5 interventions; prior independent-reconstruction inputs required. |
| Appendix D, Qwen3 rejection/fallback | Historical Qwen3 probe | Full rejected-pair and 100-page fallback workflow not supplied here. |
| Appendix E.1, native/common quality | `experiments/final_pass/quality_timing.py` | Complete-split result and per-query records under `evidence/final-pass/quality-full/`; score/vector banks excluded. |
| Appendix E.2, generic-validation timing | Same runner | 1,800 timing records; three paired rounds over 200 pages. CPU aggregation is available. |
| Appendix F, lifecycle scaling | `experiments/paper/e2e/analyze_e2e.py` | Recorded checkpoints and lifecycle calculations; 100K/1M values remain extrapolations. |
| Appendix F, retention and cold/warm diagnostics | Separate historical probes | These additional diagnostic workflows are not included as complete runnable entries. |
| Appendix G, compression | `experiments/storage/`, `experiments/support/ir_codecs.py` | Collection-specific protocols; no full historical score banks or final figure-generation pipeline. |
| Appendix H, native Qwen3 4B / ColQwen input audits | Separate historical native-input probes | Generic eight-page Qwen3 discovery checks are included; they do not reproduce the native 100-page timing. |
| Appendix I, static census | `experiments/adapter_census/census.py` | Census runner; re-querying public repositories does not establish the exact historical census counts. |
| Appendix L, controlled pixel-budget sweep | Separate historical sweep | Complete final sweep pipeline and figure regeneration are not supplied here. |

The supplementary code covers the central implementation and multiple primary
experiment paths. Rows marked incomplete are not silently represented by other
experiments with different inputs or timing boundaries.
