# Interface comparison, retrieval quality, and validation timing

Runners for the expert-interface comparison, native/common processor quality,
and generic-validation timing. Use the configurations and scope documented in
[submission coverage](../../docs/submission-coverage.md).

## Manual versus automatic interface

`manual_interface.py` holds an independent expert interface and all named visual
parameters **and registered buffers**. Qwen2.5 has one output; Qwen3 4B has the
merged output and three DeepStack outputs (leaves 1–4 of its ModelOutput; leaf 0
is unused). Both paths use `hooked.capture`, serialization, `dependency_digest`,
and `hooked.resume`. Compare both with an independently executed full target.
Source and target dependency names are compared after the explicit wrapper-prefix
mapping for the Qwen3 pair. The prefix mapping is an architecture-specific rule.

Qwen3 uses the retained Tomoro/OpenSearch publisher wrappers and the previously
documented loader adaptations in `qwen3_loader.py`: parent loader key mapping,
unexecuted generation-head guard, target dimensions, and image modality IDs.
This experiment does not claim that integration or operator/configuration
compatibility is discovered automatically.

## Native/common quality and validation timing

`quality_timing.py` uses ColNomic 3B with its own processor for the native route. Native retains the
released 1,003,520 pixel budget; common uses the manuscript’s Qwen base processor at 602,112 pixels. The source
retains state under this exact common input recipe. Both source and target are
loaded independently. All named traced dependencies must match as loaded.

Full evaluation uses the entire ViDoRe v3 HR test corpus (1,110 pages), all 1,908
queries and its qrels. The scorer uses FP32 MaxSim, TF32 disabled, stable corpus
ID ties, and linear graded nDCG at 5 and 10, matching the official evaluator's
`pytrec_eval` gain definition. Duplicate qrel pairs use the last row, as in the
official BEIR adapter. Common full and replay use the same query bank; complete
score matrices and every page representation must agree exactly. Smoke results
use incomplete data and must never be reported as standard retrieval quality.

Timing uses 200 uniformly spaced corpus pages, three paired passes, rotating
route order between passes and reversing it on odd page indices. All paths use
the same BF16 model, batch one, attention backend, image-byte memory policy and
vector serialization/fsync. Retained-file cache is deliberately warmed outside
timing for both replay routes. This is a warm-read sample measurement, not the
historical cold-read 20,946-page lifecycle experiment.

1. Full: image bytes → decode/preprocess → target model → vector write.
2. Generic: image bytes → decode/preprocess → state read/integrity/recipe checks
   → actual visual-branch input hash → target replay → vector write.
3. Fixed: state read/integrity/recipe and frozen-image hash checks → target replay
   with saved structural inputs → vector write. No processor runs here.

Model loading, one-time target dependency hashing, tracing, warmup, and output
comparison are excluded from paired page-loop timing. Load, dependency-check
and source-trace durations are recorded separately; warmup and correctness
comparison do not have separate duration measurements.
Processed input hashing is measured inside the actual generic resume stub.
Fixed-route placeholder allocations/transfers are included. Upstream operator
semantics, non-tensor configuration and deterministic processing remain explicit
integration premises; a matching recipe identifier does not prove them.

## Execution

Use `PYTHONPATH=<deployed repository root>`, set `CUDA_VISIBLE_DEVICES` to the
recorded A100, and pass the retained JSON configuration. Run a two-page smoke
before the full invocation. Each output directory must be new; failed runs are
preserved. Qwen2.5 uses the existing 4.53.2 environment; Qwen3 uses an isolated
5.9.0 dependency overlay because its original environment is no longer present.

```sh
python experiments/final_pass/manual_interface.py --architecture qwen25 --config config.json --output manual-qwen25-smoke --pages 2
python experiments/final_pass/manual_interface.py --architecture qwen3 --config config.json --output manual-qwen3-smoke --pages 2
python experiments/final_pass/quality_timing.py --config config.json --output quality-smoke --limit 2 --timing-pages 2
python experiments/final_pass/quality_timing.py --config config.json --output quality-full --timing-pages 200
```

Keep generated artifacts outside the checkout. A configurable launch wrapper
is included. Supplied configurations, model revisions, measurements and package
inventories are described in the reproduction guide and evidence index.
