# Statistics appendix

- Experimental unit for timing: one complete 500- or 1,000-item target-index transition.
- Repetitions: three sequential complete transitions per benchmark on the same RTX 4090 host.
- Timing summary: arithmetic mean ± sample SD; raw observations are retained in `stats.json`.
- Pairing: raw and ReprForge routes are measured inside each complete run; saving is calculated per run before summarization.
- Quality: deterministic serialized-index nDCG identity was checked across repeats; no additional quality uncertainty is claimed from technical repeats.
- Gate validation: booleans are checked by name; the exact lowering error of 0.0 is not incorrectly treated as a failed Boolean.
- Multiplicity: no new hypothesis tests were performed in this technical-repeat bundle.
- Hardware: a single 24,564 MiB RTX 4090 container with 12 visible CPU cores; results are not pooled with prior A100 or other 4090 hosts.
