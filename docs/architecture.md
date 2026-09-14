# Architecture

ReprForge reconstructs a target retriever's document representations from
retained source computation. The system separates state capture, validation,
and target execution so each model integration has an explicit replay contract.

## Replayable state

The ColQwen2.5 integration retains post-merger visual tokens after inverse window
reordering, together with token IDs, attention masks and image-grid metadata.
The target reconstructs text embeddings and multimodal positions, then executes
its language model, retrieval projection, normalization and output masking.

| State field | Contents |
|---|---|
| `vision` | BF16 visual tokens, `[visual_tokens, hidden_size]` |
| `input_ids` | Token IDs defining the input sequence |
| `attention_mask` | Valid sequence positions |
| `image_grid_thw` | Visual grid dimensions |

Keeping source text embeddings out of the artifact allows target embedding
changes without invalidating unchanged visual computation. The saved token IDs
still depend on the tokenizer and input template.

## Dependency validation

`VersionManifest` records component identities. A retained state's declared
dependencies are compared with the source-to-target component changes. An
unresolved change in a required component rejects reuse and selects raw encoding.

| Changed component | Post-merger visual state |
|---|---|
| Decoder adapter or retrieval projection | Reusable if the remaining contract matches |
| Text embedding weights | Target reconstructs embeddings from retained IDs |
| Processor, tokenizer or input template | Requires equivalent processed inputs |
| Vision encoder or merger | Requires equivalent upstream computation |

Component identities and dependency declarations are supplied by the integration.
The library checks those declarations; it does not automatically discover the
complete forward graph. The real-model runner establishes a pinned contract
before comparing independently executed raw and replay paths.

## Target reconstruction

For a page `d`, write target encoding as `D_t(d) = R_t(S_t(d))`. Reuse is exact
when the retained state equals `S_t(d)` and continuation faithfully executes
`R_t`. An unchanged state alone is insufficient if reconstruction omits a mask,
position or target-side operation.

INT8 and PCA preserve the validity requirement but allow numerical distortion
in the stored activation. Their evaluation reports both retrieval quality and
target agreement. Structural metadata remains part of the recovery contract.

## Implementation map

| Component | Implementation |
|---|---|
| State and encoder interface | [`adapter.py`](../reprforge/adapter.py) |
| Component versions and changes | [`versions.py`](../reprforge/versions.py), [`dependencies.py`](../reprforge/dependencies.py) |
| Replay eligibility and fallback routing | [`planning.py`](../reprforge/planning.py) |
| Scoped processor equivalence | [`equivalence.py`](../reprforge/equivalence.py) |
| MaxSim, agreement and reference index | [`index.py`](../reprforge/index.py) |
| Generation persistence | [`generation.py`](../reprforge/generation.py) |
| ColQwen2.5 capture and continuation | [`integrations/colqwen.py`](../reprforge/integrations/colqwen.py) |

The NumPy package provides the control and reference-index components. PyTorch
model execution is shared by the command-line tools and experiment runners. The repository does not
provide a universal checkpoint converter or a distributed serving backend.
