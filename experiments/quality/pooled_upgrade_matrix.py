#!/usr/bin/env python3
"""Pooled-gallery upgrade matrix: one deployed ColQwen2.5 adapter, four public targets.

Every page of six ViDoRe collections plus MMDocIR distractor pages forms one
gallery. The visual prefix is computed once per page (it is shared by every
version on the same base), then replayed through the deployed suffix (stale
index), each target suffix (full re-encode), and each codec-restored cut. Query
sets are the 30% per-collection holdouts. Routes per target:

  target_full   full re-encode with the target (reference, upper bound)
  stale         legacy index, target queries (no-op baseline)
  procrustes / affine   Drift-Adapter style query bridges into the legacy space
  hot_refresh@f mixed bank: fraction f of pages re-encoded, rest stale
  <codec>       exact cut replayed through the target suffix after the codec
  pooled_*      Ward token pooling (factor 3) on target_full and pca256 banks

The protocol is frozen before any GPU output; predictions live in protocol.json.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--matrix-code-root", type=Path, required=True)
    p.add_argument("--support-code-root", type=Path, required=True)
    p.add_argument("--collections", type=Path, required=True, help="JSON: name -> {parquet, query_column}")
    p.add_argument("--distractor-parquet", type=Path, default=None, help="parquet with image_binary column")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--processor-model", type=Path, required=True)
    p.add_argument("--max-pixels", type=int, required=True)
    p.add_argument("--family", choices=("colqwen2.5", "colqwen2"), default="colqwen2.5")
    p.add_argument("--exactness-pages", type=int, default=40, help="pages on which exact-cut terminal vectors are compared bitwise with the target")
    p.add_argument("--deployed-adapter", type=Path, required=True)
    p.add_argument("--deployed-projection", type=str, required=True)
    p.add_argument("--targets", type=Path, required=True, help="JSON: name -> {adapter, projection}")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--pool-workers", type=int, default=32)
    p.add_argument("--smoke", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------- codecs

def codec_rank(codec: str) -> int | None:
    return int(codec.removeprefix("pca").split("_", 1)[0]) if codec.startswith("pca") else None


def encode_decode(values: Any, codec: str, torch: Any, mean: Any, basis: Any) -> tuple[Any, int]:
    tokens, channels = values.shape
    x = values.float()
    if codec == "bf16":
        return x.to(torch.bfloat16), tokens * channels * 2
    if codec == "int8_token":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(x / scale).clamp(-127, 127)
        return (q * scale).to(torch.bfloat16), tokens * channels + tokens * 2
    rank = codec_rank(codec)
    if rank is None:
        raise ValueError(codec)
    b = basis[:, :rank]
    coefficients = (x - mean) @ b
    scale = coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(coefficients / scale).clamp(-127, 127)
    return ((q * scale) @ b.T + mean).to(torch.bfloat16), tokens * rank + tokens * 2


# ----------------------------------------------------------------------------- metrics

def metric_rows_gold(scores: Any, gold: list[int], target_order: Any) -> tuple[dict[str, float], list[dict[str, float]], Any]:
    import numpy as np

    order = np.argsort(-scores, axis=1, kind="stable")
    rows = []
    for q in range(len(order)):
        rank = int(np.flatnonzero(order[q] == gold[q])[0]) + 1
        t10 = set(int(v) for v in target_order[q, :10]); s10 = set(int(v) for v in order[q, :10])
        t5 = set(int(v) for v in target_order[q, :5]); s5 = set(int(v) for v in order[q, :5])
        rows.append({
            "rank": rank,
            "ndcg_at_5": 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0,
            "recall_at_1": float(rank <= 1),
            "recall_at_5": float(rank <= 5),
            "recall_at_10": float(rank <= 10),
            "reciprocal_rank": 1.0 / rank,
            "top10_overlap": len(t10 & s10) / 10.0,
            "top5_overlap": len(t5 & s5) / 5.0,
            "top1_agreement": float(order[q, 0] == target_order[q, 0]),
        })
    keys = [k for k in rows[0] if k != "rank"]
    aggregate = {k: statistics.fmean(r[k] for r in rows) for k in keys}
    aggregate["top1_flip_rate"] = 1.0 - aggregate["top1_agreement"]
    return aggregate, rows, order[:, :20]


def ward_pool(args: tuple[Any, int]) -> Any:
    """Hierarchical token pooling (Clavie et al. 2024): Ward linkage, n//factor clusters, mean, renormalise."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    x, factor = args
    n = x.shape[0]
    k = max(1, n // factor)
    if n <= 2 or k >= n:
        return x
    labels = fcluster(linkage(x, method="ward"), t=k, criterion="maxclust")
    out = np.zeros((labels.max(), x.shape[1]), dtype=np.float32)
    counts = np.zeros(labels.max(), dtype=np.float32)
    np.add.at(out, labels - 1, x)
    np.add.at(counts, labels - 1, 1.0)
    out /= counts[:, None]
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-8)
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    import numpy as np
    import torch
    import torch.nn.functional as F
    if args.family == "colqwen2.5":
        from colpali_engine.models import ColQwen2_5 as ModelClass, ColQwen2_5_Processor as ProcessorClass
    else:
        from colpali_engine.models import ColQwen2 as ModelClass, ColQwen2Processor as ProcessorClass

    sys.path.insert(0, str(args.matrix_code_root)); sys.path.insert(0, str(args.support_code_root))
    from run_matrix import ParquetRows, paired_bootstrap, sha256
    from run_smoke import load_projection as load_projection_header
    from streaming_maxsim import streaming_maxsim

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen-before-gpu-output":
        raise RuntimeError("protocol not frozen")
    torch.manual_seed(int(protocol["torch_seed"]))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)  # torch 2.5.0 cuDNN SDPA emits NaN in the Qwen2.5-VL vision tower

    # ------------------------------------------------------------------ models
    def load_projection(spec: str) -> dict[str, Any]:
        path = Path(spec)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            t = load_file(str(path))
            return {"custom_text_proj.weight": t["custom_text_proj.weight"], "custom_text_proj.bias": t["custom_text_proj.bias"]}
        if path.suffix == ".json":
            d = json.loads(path.read_text())["tensors"]
            dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
            return {k: torch.frombuffer(bytearray(bytes.fromhex(v["bytes"])), dtype=dt[v["dtype"]]).reshape(v["shape"]).clone()
                    for k, v in d.items()}
        return load_projection_header(path, torch)

    # One shared base model; every version is a named LoRA adapter plus its own projection head.
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    base = ModelClass.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    base = base.to(args.device).eval()
    peft_model: Any = None
    projections: dict[str, dict[str, Any]] = {}
    adapter_diagnostics: dict[str, Any] = {}
    import re

    linear_names = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]  # before any injection

    def resolve_targets(config: Any) -> list[str]:
        """Turn a vendor's regex/list target spec into explicit module names on the pristine base.

        PEFT matches a regex against every module name including the ModuleDicts it injected for
        earlier adapters, so a second adapter with a broad regex would try to wrap `lora_dropout`.
        """
        spec = config.target_modules
        if isinstance(spec, str):
            return [n for n in linear_names if re.fullmatch(spec, n)]
        return [n for n in linear_names if any(n == t or n.endswith("." + t) for t in spec)]

    def adapter_id(name: str) -> str:
        return "v_" + name.replace(".", "_").replace("-", "_")  # torch module names cannot contain "."

    def add_version(name: str, adapter: Path, projection: dict[str, Any]) -> None:
        nonlocal peft_model
        config = PeftConfig.from_pretrained(adapter, local_files_only=True)
        targets_resolved = resolve_targets(config)
        if not targets_resolved:
            raise RuntimeError(f"no target modules resolved for {name}: {config.target_modules}")
        config.target_modules = targets_resolved
        if peft_model is None:
            peft_model = get_peft_model(base, config, adapter_name=adapter_id(name))
        else:
            peft_model.add_adapter(adapter_id(name), config)
        weights = load_peft_weights(adapter, device="cpu", local_files_only=True)
        old, new = "base_model.model.model.layers.", "base_model.model.language_model.layers."
        remapped = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in weights.items()}
        res = set_peft_model_state_dict(peft_model, remapped, adapter_name=adapter_id(name))
        missing = [k for k in res.missing_keys if "lora_" in k and f".{adapter_id(name)}." in k]
        unexpected = [k for k in res.unexpected_keys if "lora_" in k]
        if missing or unexpected:
            raise RuntimeError(f"adapter load incomplete for {name}: missing={len(missing)} unexpected={len(unexpected)} e.g. {(missing + unexpected)[:2]}")
        adapter_diagnostics[name] = {"checkpoint_tensors": len(weights), "remapped": sum(k.startswith(old) for k in weights),
                                     "vision_tensors": sum("visual" in k for k in weights), "target_modules": len(targets_resolved),
                                     "lora_r": config.r, "lora_alpha": config.lora_alpha}
        projections[name] = {k: v.to(args.device) for k, v in projection.items()}
        peft_model.to(args.device)

    def activate(name: str) -> None:
        peft_model.set_adapter(adapter_id(name))
        with torch.no_grad():
            base.custom_text_proj.weight.copy_(projections[name]["custom_text_proj.weight"].to(base.custom_text_proj.weight))
            base.custom_text_proj.bias.copy_(projections[name]["custom_text_proj.bias"].to(base.custom_text_proj.bias))

    # pin artifacts to the protocol
    pins = {"deployed": (args.deployed_adapter, args.deployed_projection, protocol["deployed"])}
    targets_spec = json.loads(args.targets.read_text())
    for name, spec in targets_spec.items():
        pins[name] = (Path(spec["adapter"]), spec["projection"], protocol["targets"][name])
    for name, (adapter, projection, pin) in pins.items():
        if sha256(adapter / "adapter_model.safetensors") != pin["adapter_sha256"]:
            raise RuntimeError(f"adapter hash mismatch: {name}")
        if sha256(Path(projection)) != pin["projection_sha256"]:
            raise RuntimeError(f"projection hash mismatch: {name}")
        add_version(name, adapter, load_projection(str(projection)))
        print(json.dumps({"loaded": name, **adapter_diagnostics[name]}), flush=True)
    for prm in peft_model.parameters():
        prm.requires_grad_(False)
    peft_model.eval()
    target_names = list(targets_spec)
    codecs_of = {name: list(protocol["targets"][name]["codecs"]) for name in target_names}
    all_codecs = sorted({c for cs in codecs_of.values() for c in cs}, key=lambda c: (codec_rank(c) or 10_000))
    max_rank = max([r for r in (codec_rank(c) for c in all_codecs) if r is not None] + [1])

    processor = ProcessorClass.from_pretrained(args.processor_model, local_files_only=True)
    processor.image_processor.max_pixels = int(args.max_pixels)
    processor.image_processor.size["longest_edge"] = int(args.max_pixels)

    def sync() -> None:
        torch.cuda.synchronize()

    def prefix(base: Any, batch: Any) -> dict[str, Any]:
        input_ids, attention, grid = batch["input_ids"], batch["attention_mask"], batch["image_grid_thw"]
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat([row[: int(o.item())] for row, o in zip(batch["pixel_values"], offsets, strict=True)], dim=0)
        emb = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(emb.device, emb.dtype)
        if not torch.isfinite(vision).all():
            raise RuntimeError("non-finite vision features")
        emb = emb.masked_scatter((input_ids == int(base.config.image_token_id)).unsqueeze(-1).expand_as(emb), vision)
        pos, _ = base.get_rope_index(input_ids, grid, None, attention_mask=attention)
        return {"attention": attention, "embeds": emb, "position_ids": pos,
                "cache_position": torch.arange(emb.shape[1], device=args.device),
                "valid": torch.where(attention[0].bool())[0],
                "visual": torch.where((input_ids[0] == int(base.config.image_token_id)) & attention[0].bool())[0]}

    def suffix(base: Any, data: dict[str, Any], embeds: Any) -> Any:
        out = base.language_model(input_ids=None, inputs_embeds=embeds, attention_mask=data["attention"], position_ids=data["position_ids"],
                                  cache_position=data["cache_position"], use_cache=False, output_hidden_states=False, return_dict=True)
        proj = F.normalize(base.custom_text_proj(out.last_hidden_state), dim=-1)
        return proj[0].index_select(0, data["valid"])

    # ------------------------------------------------------------------ gallery
    collections_spec = json.loads(args.collections.read_text())
    datasets = {name: ParquetRows(Path(spec["parquet"]), spec["query_column"]) for name, spec in collections_spec.items()}
    import hashlib
    import pyarrow.parquet as pq

    def image_hashes(path: Path, column: str) -> list[str]:
        col = pq.read_table(path, columns=[column]).column(column).to_pylist()
        out = []
        for v in col:
            blob = v.get("bytes") if isinstance(v, dict) else v
            out.append(hashlib.sha1(blob).hexdigest())
        return out

    # ViDoRe test sets repeat one page image across several queries (TAT-DQA: 1,663 rows on 271 pages).
    # The gallery holds each distinct image once; every query's gold is that page's single gallery slot.
    gallery: list[tuple[str, int]] = []          # first (collection, row) that introduced each distinct image
    position: dict[tuple[str, int], int] = {}    # (collection, row) -> gallery slot
    slot_of_hash: dict[str, int] = {}
    duplicates = {}
    query_rows: dict[str, list[int]] = {}
    fit_rows: dict[str, list[int]] = {}
    calibration_rows: dict[str, list[int]] = {}
    for name, ds in datasets.items():
        n = len(ds)
        hashes = image_hashes(Path(collections_spec[name]["parquet"]), "image")
        perm = np.random.RandomState(0).permutation(n)
        holdout = sorted(int(v) for v in perm[: int(round(0.3 * n))])
        hs = set(holdout)
        non_holdout = [i for i in range(n) if i not in hs]
        if args.smoke:
            holdout = holdout[: args.smoke]
            non_holdout = non_holdout[: args.smoke]
        def has_query(i: int) -> bool:  # ViDoRe synthetic sets carry queries on a subset of rows only
            q = ds.table.slice(i, 1).to_pylist()[0][collections_spec[name]["query_column"]]
            return bool(q) and str(q).strip().lower() not in ("none", "nan", "")
        query_rows[name] = [i for i in holdout if has_query(i)]
        fit_rows[name] = [i for i in non_holdout if has_query(i)][:200] if not args.smoke else non_holdout[: max(4, args.smoke)]
        calibration_rows[name] = non_holdout[:16] if not args.smoke else non_holdout[: max(2, min(4, args.smoke))]
        pages = sorted(set(holdout) | set(non_holdout)) if args.smoke else list(range(n))
        dup = 0
        for i in pages:
            h = hashes[i]
            if h in slot_of_hash:
                position[(name, i)] = slot_of_hash[h]; dup += 1
            else:
                slot_of_hash[h] = len(gallery); position[(name, i)] = len(gallery); gallery.append((name, i))
        duplicates[name] = {"rows": len(pages), "distinct_pages": len(pages) - dup}
    distractor_table = None
    if args.distractor_parquet is not None:
        distractor_table = pq.read_table(args.distractor_parquet, columns=["image_binary"])
        dh = image_hashes(args.distractor_parquet, "image_binary")
        count = len(distractor_table) if not args.smoke else min(len(distractor_table), args.smoke)
        dup = 0
        for i in range(count):
            if dh[i] in slot_of_hash:
                dup += 1; continue
            slot_of_hash[dh[i]] = len(gallery); gallery.append(("distractor", i))
        duplicates["distractor"] = {"rows": count, "distinct_pages": count - dup}

    from PIL import Image

    def page_image(key: tuple[str, int]) -> Any:
        name, row = key
        if name == "distractor":
            blob = distractor_table.slice(row, 1).to_pylist()[0]["image_binary"]
            return Image.open(io.BytesIO(blob)).convert("RGB")
        return datasets[name].get(row)[0]

    print(json.dumps({"gallery_pages": len(gallery), "dedup": duplicates, "queries": {k: len(v) for k, v in query_rows.items()}}), flush=True)

    # ------------------------------------------------------------------ calibration (query-free, pooled)
    cal_started = time.perf_counter()
    observed = []
    with torch.inference_mode():
        for name, rows in calibration_rows.items():
            for i in rows:
                activate("deployed")
                data = prefix(base, processor.process_images([datasets[name].get(i)[0]]).to(args.device))
                observed.append(data["embeds"][0].index_select(0, data["visual"]).detach().cpu())
    observed_t = torch.cat(observed, dim=0)
    token_count = min(int(protocol["sampled_visual_tokens"]), len(observed_t))
    selection = torch.linspace(0, len(observed_t) - 1, token_count).round().long()
    cal = observed_t.index_select(0, selection).to(args.device).float()
    mean = cal.mean(dim=0)
    rank = min(max_rank, len(cal) - 1)
    centered = (cal - mean).double()
    covariance = (centered.T @ centered).cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eig_order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[eig_order], eigenvectors[:, eig_order]
    basis = eigenvectors[:, :rank].float().to(args.device)
    sync()
    calibration_seconds = time.perf_counter() - cal_started
    explained = eigenvalues.clamp_min(0).cumsum(0) / eigenvalues.clamp_min(0).sum()
    torch.save({"mean": mean.to(torch.bfloat16).cpu(), "basis": basis.to(torch.bfloat16).cpu()}, args.output_root / "basis.pt")

    # ------------------------------------------------------------------ canary: identical prefixes across all versions
    first = processor.process_images([page_image(gallery[0])]).to(args.device)
    with torch.inference_mode():
        activate("deployed")
        ref = prefix(base, first)["embeds"]
        for name in target_names:
            activate(name)
            if not torch.equal(ref, prefix(base, first)["embeds"]):
                raise RuntimeError(f"prefix canary failed for {name}")

    # ------------------------------------------------------------------ encode gallery
    banks: dict[str, list[Any]] = {"stale": []}
    for name in target_names:
        banks[f"{name}/target_full"] = []
        for c in codecs_of[name]:
            banks[f"{name}/{c}"] = []
    stage = {"preprocess": 0.0, "prefix": 0.0, "suffix_deployed": 0.0, **{f"suffix_target/{n}": 0.0 for n in target_names}}
    replay_seconds = {f"{n}/{c}": 0.0 for n in target_names for c in codecs_of[n]}
    payload_bytes = {c: 0 for c in all_codecs}
    tokens = {"visual": 0, "sequence": 0, "terminal": 0}
    per_page: list[dict[str, Any]] = []
    exactness: dict[str, dict[str, Any]] = {n: {"pages": 0, "max_abs_error": 0.0, "bitwise_equal_pages": 0, "bitwise_equal_elements": 0, "elements": 0} for n in target_names if "bf16" in codecs_of[n]}
    encode_started = time.perf_counter()
    with torch.inference_mode():
        for idx, key in enumerate(gallery):
            image = page_image(key)
            activate("deployed")
            sync(); t0 = time.perf_counter()
            batch = processor.process_images([image]).to(args.device)
            sync(); t1 = time.perf_counter()
            data = prefix(base, batch)
            sync(); t2 = time.perf_counter()
            stale = suffix(base, data, data["embeds"])
            sync(); t3 = time.perf_counter()
            stage["preprocess"] += t1 - t0; stage["prefix"] += t2 - t1; stage["suffix_deployed"] += t3 - t2
            banks["stale"].append(stale.to(torch.bfloat16).cpu())
            visual = data["visual"]
            restored_cache: dict[str, Any] = {}
            for name in target_names:
                activate(name)
                sync(); s0 = time.perf_counter()
                full = suffix(base, data, data["embeds"])
                sync(); s1 = time.perf_counter()
                stage[f"suffix_target/{name}"] += s1 - s0
                banks[f"{name}/target_full"].append(full.to(torch.bfloat16).cpu())
                for codec in codecs_of[name]:
                    sync(); r0 = time.perf_counter()
                    if codec not in restored_cache:
                        restored, nbytes = encode_decode(data["embeds"][0].index_select(0, visual), codec, torch, mean, basis)
                        restored_cache[codec] = restored
                        payload_bytes[codec] += nbytes
                    embeds = data["embeds"].clone()
                    embeds[0, visual] = restored_cache[codec]
                    replayed = suffix(base, data, embeds)
                    sync()
                    replay_seconds[f"{name}/{codec}"] += time.perf_counter() - r0
                    banks[f"{name}/{codec}"].append(replayed.to(torch.bfloat16).cpu())
                    if codec == "bf16" and idx < args.exactness_pages:
                        a = replayed.to(torch.bfloat16); b = full.to(torch.bfloat16)
                        e = exactness[name]; e["pages"] += 1
                        e["max_abs_error"] = max(e["max_abs_error"], float((a.float() - b.float()).abs().max()))
                        eq = int((a == b).sum()); e["bitwise_equal_elements"] += eq; e["elements"] += a.numel()
                        e["bitwise_equal_pages"] += int(eq == a.numel())
            nvis, nseq, nterm = len(visual), int(data["attention"][0].sum()), len(data["valid"])
            tokens["visual"] += nvis; tokens["sequence"] += nseq; tokens["terminal"] += nterm
            per_page.append({"gallery_id": idx, "collection": key[0], "row": key[1], "visual_tokens": nvis, "sequence_tokens": nseq,
                             "preprocess_s": t1 - t0, "prefix_s": t2 - t1, "suffix_deployed_s": t3 - t2})
            if idx % 25 == 0 or idx == len(gallery) - 1:
                elapsed = time.perf_counter() - encode_started
                print(json.dumps({"encoded": idx + 1, "total": len(gallery), "elapsed_s": round(elapsed), "eta_s": round(elapsed / (idx + 1) * (len(gallery) - idx - 1))}), flush=True)
    pages = len(gallery)
    suffix_share = {n: stage[f"suffix_target/{n}"] / (stage["preprocess"] + stage["prefix"] + stage[f"suffix_target/{n}"]) for n in target_names}

    # ------------------------------------------------------------------ queries
    def encode_queries(version: str, texts: list[str]) -> list[Any]:
        activate(version)
        model = peft_model
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), args.query_batch_size):
                qb = processor.process_queries(texts[start: start + args.query_batch_size]).to(args.device)
                vec = model(**qb).float()
                for v, mask in zip(vec, qb["attention_mask"], strict=True):
                    out.append(v[mask.bool()].detach().cpu())
        return out

    eval_texts, gold, eval_collection = [], [], []
    for name, rows in query_rows.items():
        for i in rows:
            eval_texts.append(datasets[name].get(i)[1]); gold.append(position[(name, i)]); eval_collection.append(name)
    fit_texts = [datasets[name].get(i)[1] for name, rows in fit_rows.items() for i in rows]
    deployed_fit = encode_queries("deployed", fit_texts)
    y = torch.cat(deployed_fit, dim=0).to(args.device).double()

    # ------------------------------------------------------------------ token pooling (CPU, parallel) for selected banks
    from multiprocessing import get_context

    pool_factor = 3
    pooled_targets = [n for n in target_names if n in ("vidore-v0.2", "metric-ai-3b")] or target_names[:2]
    pooled_banks: dict[str, list[Any]] = {}
    pool_started = time.perf_counter()
    with get_context("spawn").Pool(args.pool_workers) as workers:
        for n in pooled_targets:
            for route in ("target_full", "pca256_int8"):
                src = banks[f"{n}/{route}"]
                arrays = [v.float().numpy() for v in src]
                pooled = workers.map(ward_pool, [(a, pool_factor) for a in arrays], chunksize=8)
                pooled_banks[f"{n}/pooled_{route}"] = [torch.from_numpy(p).to(torch.bfloat16) for p in pooled]
    pooling_seconds = time.perf_counter() - pool_started

    # ------------------------------------------------------------------ scoring per target
    hot_fractions = {"matched": None, "0.25": 0.25, "0.5": 0.5}
    hot_perm = np.random.RandomState(1).permutation(pages)
    results: dict[str, Any] = {}
    rankings: dict[str, Any] = {}
    per_query_out: dict[str, Any] = {}
    for tname in target_names:
        tq = encode_queries(tname, eval_texts)
        tfit = encode_queries(tname, fit_texts)
        x = torch.cat(tfit, dim=0).to(args.device).double()
        u, _, vt = torch.linalg.svd(x.T @ y)
        procrustes = (u @ vt).float().cpu()
        xa = torch.cat((x, torch.ones((x.shape[0], 1), device=args.device, dtype=x.dtype)), dim=1)
        affine = torch.linalg.solve(xa.T @ xa + 0.01 * torch.eye(xa.shape[1], device=args.device, dtype=x.dtype), xa.T @ y).float().cpu()

        def bridge(values: list[Any], kind: str) -> list[Any]:
            out = []
            for v in values:
                src = v.float()
                mapped = src @ procrustes if kind == "procrustes" else torch.cat((src, torch.ones((src.shape[0], 1))), dim=1) @ affine
                out.append(F.normalize(mapped, dim=-1))
            return out

        routes: dict[str, tuple[list[Any], list[Any]]] = {  # route -> (queries, documents)
            "target_full": (tq, banks[f"{tname}/target_full"]),
            "stale": (tq, banks["stale"]),
            "procrustes_bridge": (bridge(tq, "procrustes"), banks["stale"]),
            "affine_bridge": (bridge(tq, "affine"), banks["stale"]),
        }
        for label, f in hot_fractions.items():
            frac = suffix_share[tname] if f is None else f
            refreshed = set(int(v) for v in hot_perm[: int(round(frac * pages))])
            mixed = [banks[f"{tname}/target_full"][i] if i in refreshed else banks["stale"][i] for i in range(pages)]
            routes[f"hot_refresh@{label}"] = (tq, mixed)
        for c in codecs_of[tname]:
            routes[c] = (tq, banks[f"{tname}/{c}"])
        if tname in pooled_targets:
            routes["pooled_target_full"] = (tq, pooled_banks[f"{tname}/pooled_target_full"])
            routes["pooled_pca256_int8"] = (tq, pooled_banks[f"{tname}/pooled_pca256_int8"])

        scores = {r: streaming_maxsim(q, d, device=args.device, query_chunk=32, document_chunk=16) for r, (q, d) in routes.items()}
        target_order = np.argsort(-scores["target_full"], axis=1, kind="stable")
        pooled_order = np.argsort(-scores["pooled_target_full"], axis=1, kind="stable") if "pooled_target_full" in scores else None
        aggregates, rows_by_route, top20 = {}, {}, {}
        for r in routes:
            reference = pooled_order if (r == "pooled_pca256_int8" and pooled_order is not None) else target_order
            aggregates[r], rows_by_route[r], top20[r] = metric_rows_gold(scores[r], gold, reference)
        # per-collection breakdown
        by_collection: dict[str, dict[str, dict[str, float]]] = {}
        for r in routes:
            by_collection[r] = {}
            for cname in query_rows:
                sel = [row for row, c in zip(rows_by_route[r], eval_collection, strict=True) if c == cname]
                by_collection[r][cname] = {k: statistics.fmean(s[k] for s in sel) for k in sel[0] if k != "rank"}
        route_names = list(routes)
        bootstrap = {
            r: {f: paired_bootstrap(rows_by_route[r], rows_by_route["target_full"], field=f, draws=20_000,
                                    seed=int(protocol["torch_seed"]) + 1000 * target_names.index(tname) + 10 * route_names.index(r) + fi)
                for fi, f in enumerate(("ndcg_at_5", "top10_overlap", "recall_at_1"))}
            for r in route_names if r != "target_full"
        }
        results[tname] = {
            "kind": protocol["targets"][tname]["kind"],
            "suffix_share_matched_fraction": suffix_share[tname],
            "hot_refresh_fractions": {k: (suffix_share[tname] if v is None else v) for k, v in hot_fractions.items()},
            "bridge_fit": {"queries": len(fit_texts), "token_pairs": int(x.shape[0]), "ridge_lambda": 0.01},
            "metrics": aggregates,
            "metrics_by_collection": by_collection,
            "paired_bootstrap_vs_target_full": bootstrap,
        }
        rankings[tname] = {r: top20[r].tolist() for r in routes}
        per_query_out[tname] = {r: rows_by_route[r] for r in routes}
        print(json.dumps({"scored": tname, "ndcg5": {r: round(aggregates[r]["ndcg_at_5"], 4) for r in routes},
                          "ta10": {r: round(aggregates[r]["top10_overlap"], 4) for r in routes}}), flush=True)

    terminal_dim = int(banks["stale"][0].shape[-1])
    summary = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha256(args.protocol), "smoke": args.smoke,
        "hardware": torch.cuda.get_device_name(torch.cuda.current_device()),
        "processor": {"max_pixels": int(args.max_pixels), "min_pixels": int(processor.image_processor.size["shortest_edge"])},
        "gallery": {"pages": pages, "by_collection": {c: sum(1 for k in gallery if k[0] == c) for c in dict.fromkeys(k[0] for k in gallery)}, "dedup": duplicates,
                    "queries": {k: len(v) for k, v in query_rows.items()}, "queries_total": len(eval_texts)},
        "tokens_per_page": {k: v / pages for k, v in tokens.items()} | {"hidden_channels": int(mean.shape[0]), "terminal_dim": terminal_dim},
        "stage_seconds_total": stage,
        "stage_seconds_per_page": {k: v / pages for k, v in stage.items()},
        "cut_leverage": {n: 1.0 - suffix_share[n] for n in target_names},
        "exact_cut_terminal_exactness": exactness,
        "family": args.family,
        "replay_seconds_per_page": {k: v / pages for k, v in replay_seconds.items()},
        "payload_bytes_per_page": {c: v / pages for c, v in payload_bytes.items()},
        "terminal_bytes_per_page": {"float32": tokens["terminal"] / pages * terminal_dim * 4, "bf16": tokens["terminal"] / pages * terminal_dim * 2,
                                    "pooled3_bf16": tokens["terminal"] / pages / pool_factor * terminal_dim * 2,
                                    "light10_sq8": tokens["terminal"] / pages * 0.10 * terminal_dim},
        "basis_bytes": (args.output_root / "basis.pt").stat().st_size,
        "calibration_seconds": calibration_seconds, "pooling_seconds": pooling_seconds, "pool_factor": pool_factor,
        "pca_explained_variance": {str(r): float(explained[r - 1]) for r in (64, 128, 256, 512) if r <= rank},
        "targets": results,
    }
    (args.output_root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "rankings.json").write_text(json.dumps({"gold": gold, "collection": eval_collection, "gallery": gallery, "top20": rankings}) + "\n")
    (args.output_root / "per_query.json").write_text(json.dumps(per_query_out) + "\n")
    (args.output_root / "per_page.json").write_text(json.dumps(per_page) + "\n")
    print(json.dumps({"done": str(args.output_root), "cut_leverage": summary["cut_leverage"]}), flush=True)


if __name__ == "__main__":
    main()
