"""Census of which encoder stages public ColPali-family releases actually change.

Reads only safetensors headers and small JSON files from the Hugging Face Hub;
no model weights are downloaded. For every repository it records tensor names
and classifies them into stages of the document-encoding path:

  vision      -- visual tower blocks (ViT / SigLIP)
  merger      -- vision->language projection (Qwen2-VL merger, PaliGemma
                 multi_modal_projector)
  decoder     -- language-model transformer layers / embeddings
  terminal    -- retrieval projection head (custom_text_proj / linear head)
  other       -- anything else (lm_head, norms not attributable, etc.)

For LoRA adapters the classification is applied to the module path that the
LoRA tensors wrap, so an adapter that only touches decoder layers reports zero
vision tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

REPOS = [
    # (repo_id, family, kind, note)
    ("vidore/colpali", "colpali", "adapter", "v1.0 release"),
    ("vidore/colpali-v1.1", "colpali", "adapter", "v1.1 release"),
    ("vidore/colpali-v1.2", "colpali", "adapter", "v1.2 release"),
    ("vidore/colpali-v1.3", "colpali", "adapter", "v1.3 release"),
    ("vidore/colqwen2-v0.1", "colqwen2", "adapter", "v0.1 release"),
    ("vidore/colqwen2-v1.0", "colqwen2", "adapter", "v1.0 release"),
    ("vidore/colqwen2.5-v0.1", "colqwen2.5", "adapter", "v0.1 release"),
    ("vidore/colqwen2.5-v0.2", "colqwen2.5", "adapter", "v0.2 release"),
    ("vidore/colSmol-256M", "colsmol", "adapter", "256M release"),
    ("vidore/colSmol-500M", "colsmol", "adapter", "500M release"),
    ("Metric-AI/ColQwen2.5-3b-multilingual-v1.0", "colqwen2.5", "adapter", "third-party multilingual"),
    ("Metric-AI/ColQwen2.5-7b-multilingual-v1.0", "colqwen2.5-7b", "adapter", "third-party multilingual"),
    ("tsystems/colqwen2.5-3b-multilingual-v1.0", "colqwen2.5", "adapter", "third-party multilingual"),
    ("nomic-ai/colnomic-embed-multimodal-3b", "colqwen2.5", "merged", "third-party full checkpoint"),
    ("nomic-ai/colnomic-embed-multimodal-7b", "colqwen2.5-7b", "merged", "third-party full checkpoint"),
    ("vidore/colqwen2.5-v0.2-merged", "colqwen2.5", "merged", "merged release"),
    ("vidore/colpali-v1.2-merged", "colpali", "merged", "merged release"),
    ("vidore/colqwen2.5-vietnamese-1.0", "colqwen2.5", "adapter", "domain adapter"),
    ("TabularAI/colqwen2.5-v0.2-turkish", "colqwen2.5", "adapter", "domain adapter"),
    ("selimc/colpali-turkish", "colpali", "adapter", "domain adapter"),
    ("vidore/colSmol-256M-base", "colsmol", "base", "base checkpoint"),
    ("vidore/colSmol-500M-base", "colsmol", "base", "base checkpoint"),
    ("vidore/colqwen2-v1.0-merged", "colqwen2", "merged", "merged release"),
    ("vidore/colpali-v1.3-merged", "colpali", "merged", "merged release"),
    ("ModernVBERT/colmodernvbert", "modernvbert", "merged", "different family"),
    ("jinaai/jina-embeddings-v4", "jina-v4", "merged", "multi-adapter single base"),
    ("vidore/colqwen2.5-base", "colqwen2.5", "base", "base checkpoint"),
    ("vidore/colqwen2-base", "colqwen2", "base", "base checkpoint"),
    ("vidore/colpaligemma-3b-pt-448-base", "colpali", "base", "base checkpoint (pt)"),
    ("vidore/colpaligemma-3b-mix-448-base", "colpali", "base", "base checkpoint (mix)"),
]

STAGE_RULES = [
    ("vision", re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder)\.(?!merger)")),
    ("merger", re.compile(r"(^|\.)(visual\.merger|multi_modal_projector|mm_projector|connector|merger)\.")),
    ("terminal", re.compile(r"(^|\.)(custom_text_proj|linear|projection_head|score|dense_pool|single_vector_projector|multi_vector_projector)\.")),
    ("decoder", re.compile(r"(^|\.)(language_model|model\.layers|layers\.\d+|embed_tokens|text_model|model\.norm|norm)\.")),
]


def classify(name: str) -> str:
    for stage, pattern in STAGE_RULES:
        if pattern.search(name):
            return stage
    if name.endswith("lm_head.weight") or "lm_head" in name:
        return "other"
    return "other"


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def small_json(repo: str, filename: str, api_token: str | None):
    try:
        path = hf_hub_download(repo, filename, token=api_token)
    except EntryNotFoundError:
        return None, None
    return json.loads(Path(path).read_text()), sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    api = HfApi(token=args.token)
    rows = []
    for repo, family, kind, note in REPOS:
        record = {"repo": repo, "family": family, "kind": kind, "note": note}
        try:
            info = api.model_info(repo, files_metadata=False)
            record["sha"] = info.sha
            record["last_modified"] = str(info.last_modified)
            files = [f for f in api.list_repo_files(repo) if f.endswith(".safetensors")]
            if not files:
                raise RuntimeError("no safetensors files in repository")
            parsed = {f: api.parse_safetensors_file_metadata(repo, f, token=args.token) for f in files}
            record["safetensors_files"] = files
        except (RepositoryNotFoundError, GatedRepoError) as exc:
            record["error"] = type(exc).__name__
            rows.append(record)
            print(json.dumps(record), flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(record)
            print(json.dumps(record), flush=True)
            continue
        stage_counts: dict[str, int] = {}
        stage_params: dict[str, int] = {}
        lora_stage_counts: dict[str, int] = {}
        total_params = 0
        names = []
        # meta.files_metadata: {filename: SafetensorsFileMetadata(tensors={name: TensorInfo})}
        for filename, file_meta in parsed.items():
            for name, tensor in file_meta.tensors.items():
                names.append(name)
                stage = classify(name)
                numel = 1
                for dim in tensor.shape:
                    numel *= dim
                total_params += numel
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                stage_params[stage] = stage_params.get(stage, 0) + numel
                if "lora_" in name:
                    lora_stage_counts[stage] = lora_stage_counts.get(stage, 0) + 1
                per_file = record.setdefault("per_file_stage_counts", {}).setdefault(filename, {})
                per_file[stage] = per_file.get(stage, 0) + 1
        record.update(
            {
                "tensor_count": len(names),
                "total_params": total_params,
                "stage_tensor_counts": stage_counts,
                "stage_param_counts": stage_params,
                "lora_stage_tensor_counts": lora_stage_counts,
                "unclassified_examples": [n for n in names if classify(n) == "other"][:8],
            }
        )
        adapter_cfg, adapter_cfg_sha = small_json(repo, "adapter_config.json", args.token)
        if adapter_cfg:
            record["adapter_config"] = {
                "base_model_name_or_path": adapter_cfg.get("base_model_name_or_path"),
                "r": adapter_cfg.get("r"),
                "lora_alpha": adapter_cfg.get("lora_alpha"),
                "target_modules": adapter_cfg.get("target_modules"),
                "sha256": adapter_cfg_sha,
            }
        for fname in ("preprocessor_config.json", "tokenizer_config.json", "config.json"):
            data, digest = small_json(repo, fname, args.token)
            if digest:
                record[f"{fname}_sha256"] = digest
            if fname == "preprocessor_config.json" and data:
                record["preprocessor_keys"] = {
                    k: data.get(k)
                    for k in ("max_pixels", "min_pixels", "size", "image_processor_type", "processor_class")
                    if k in data
                }
            if fname == "config.json" and data:
                record["config_keys"] = {
                    k: data.get(k)
                    for k in ("architectures", "model_type", "_name_or_path", "torch_dtype")
                    if k in data
                }
        rows.append(record)
        print(json.dumps({k: v for k, v in record.items() if k in ("repo", "tensor_count", "stage_tensor_counts", "lora_stage_tensor_counts")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
