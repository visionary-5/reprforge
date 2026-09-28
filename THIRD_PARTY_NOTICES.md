# Third-party inputs

The small files under `configs/projections/` are derived from public model
inputs and are not full model checkpoints. The upstream model cards identify
these base models as Apache-2.0; the license text is supplied in `LICENSE`.
Full models downloaded separately remain subject to their respective terms.

| Included file | Upstream source | Recorded revision |
|---|---|---|
| `vidore-v0.2-base-shard2-header.bin` | [ViDoRe ColQwen2.5 base](https://huggingface.co/vidore/colqwen2.5-base) | `92908120384b7a2110c5beda3ab29cbdb2c08e49` |
| `colnomic-3b.json` | [ViDoRe ColQwen2.5 base](https://huggingface.co/vidore/colqwen2.5-base) | `92908120384b7a2110c5beda3ab29cbdb2c08e49` |
| `metric-ai-3b.json` | [Metric-AI ColQwen2.5 3B base](https://huggingface.co/Metric-AI/colqwen2.5-3b-base) | `main`, with a retained content hash |
| `tsystems-3b.json` | [T-Systems ColQwen2.5 3B base](https://huggingface.co/tsystems/colqwen2.5-3b-base) | `main`, with a retained content hash |

The binary is a retained safetensors header prefix. JSON files package the
selected projection input data. Their source filenames and SHA-256 hashes are
recorded in `experiments/reconstruction/sources.json`. Publisher identifiers
identify the upstream sources of these inputs.
