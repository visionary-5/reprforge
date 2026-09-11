#!/usr/bin/env python3
"""Independent native-raw versus persisted post-merger replay (optional GPU deps)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_model(spec, config, torch):
    from colpali_engine.models import ColQwen2_5
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from run_smoke import load_projection

    base = ColQwen2_5.from_pretrained(config["base_model"], torch_dtype=torch.bfloat16,
                                    local_files_only=True, low_cpu_mem_usage=True)
    proj_path = Path(spec["projection"])
    if proj_path.suffix == ".json":
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        proj = {k: torch.frombuffer(bytearray.fromhex(v["bytes"]), dtype=dtype[v["dtype"]])
                .reshape(v["shape"]).clone()
                for k, v in json.loads(proj_path.read_text())["tensors"].items()}
    elif proj_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        proj = load_file(str(proj_path))
    else:
        proj = load_projection(proj_path, torch)
    # Install the frozen base head BEFORE loading adapter weights, including head LoRA.
    with torch.no_grad():
        for name in ("weight", "bias"):
            value = getattr(base.custom_text_proj, name)
            value.copy_(proj["custom_text_proj." + name].to(value))
    pc = PeftConfig.from_pretrained(spec["adapter"], local_files_only=True)
    linear = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]
    target = pc.target_modules
    pc.target_modules = [n for n in linear if (re.fullmatch(target, n) if isinstance(target, str)
                         else any(n == t or n.endswith("." + t) for t in target))]
    if not pc.target_modules:
        raise RuntimeError("No adapter targets resolved")
    model = get_peft_model(base, pc)
    weights = load_peft_weights(spec["adapter"], device="cpu", local_files_only=True)
    old, new = "base_model.model.model.layers.", "base_model.model.language_model.layers."
    remapped = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in weights.items()}
    loaded = set_peft_model_state_dict(model, remapped, adapter_name="default")
    missing = [k for k in loaded.missing_keys if "lora_" in k]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(f"Incomplete adapter load: {missing}, {loaded.unexpected_keys}")
    model.to("cuda:0").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, base, {"checkpoint_tensors": len(weights), "missing_lora": missing,
                         "unexpected": loaded.unexpected_keys,
                         "mask_non_image_embeddings": base.mask_non_image_embeddings}


def capture(base, batch, torch):
    ids, grid = batch["input_ids"], batch["image_grid_thw"]
    offsets = grid[:, 1] * grid[:, 2]
    pixels = torch.cat([row[:int(o)] for row, o in zip(batch["pixel_values"], offsets)])
    vision = torch.cat(base.get_image_features(pixels, grid), dim=0)
    return {"input_ids": ids.cpu(), "attention_mask": batch["attention_mask"].cpu(),
            "image_grid_thw": grid.cpu(), "vision": vision.cpu()}


def replay(base, state, torch):
    """Recreate target text embeddings/positions; never execute target vision."""
    state = {k: v.to("cuda:0") for k, v in state.items()}
    ids, grid, mask = state["input_ids"], state["image_grid_thw"], state["attention_mask"]
    embeds = base.get_input_embeddings()(ids)
    image_mask = ids == base.config.image_token_id
    embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds),
                                  state["vision"].to(embeds))
    pos, _ = base.get_rope_index(ids, grid, None, attention_mask=mask)
    output = base.language_model(input_ids=None, inputs_embeds=embeds, attention_mask=mask,
                                 position_ids=pos, cache_position=torch.arange(ids.shape[1], device=ids.device),
                                 use_cache=False, output_hidden_states=False, return_dict=True)
    vectors = base.custom_text_proj(output.last_hidden_state)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    vectors = vectors * mask.unsqueeze(-1)
    if base.mask_non_image_embeddings:
        vectors = vectors * image_mask.unsqueeze(-1)
    return vectors[0][mask[0].bool()]


def select(config, protocol):
    import numpy as np
    import pyarrow.parquet as pq
    pools, queries, hashes = [], [], {}
    for name, spec in config["collections"].items():
        path = Path(spec["parquet"])
        hashes[name] = digest(path)
        rows = pq.read_table(path, columns=["image", spec["query_column"]]).to_pylist()
        pools.append((name, rows))
        if name != "shiftproject":
            holdout = sorted(np.random.RandomState(0).permutation(len(rows))[:int(round(.3 * len(rows)))])
            for i in holdout:
                q = rows[i][spec["query_column"]]
                if q and str(q).strip().lower() not in ("none", "nan", ""):
                    queries.append({"collection": name, "row": int(i), "text": str(q)})
    seen, selected = set(), []
    cursors = [0] * len(pools)
    while len(selected) < protocol["pages"]:
        progress = False
        for j, (name, rows) in enumerate(pools):
            while cursors[j] < len(rows):
                i = cursors[j]
                cursors[j] += 1
                v = rows[i]["image"]
                blob = v["bytes"] if isinstance(v, dict) else v
                sha = hashlib.sha256(blob).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                selected.append({"collection": name, "row": i, "image_sha256": sha})
                progress = True
                break
            if len(selected) == protocol["pages"]:
                break
        if not progress:
            raise ValueError("Insufficient unique pages")
    return selected, queries, hashes


def worker(args, config, protocol):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from PIL import Image
    from streaming_maxsim import streaming_maxsim

    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    root = args.output
    inputs = json.loads((root / "inputs.json").read_text())
    rows = {n: pq.read_table(s["parquet"], columns=["image"]).to_pylist()
            for n, s in config["collections"].items()}
    def image_of(item):
        value = rows[item["collection"]][item["row"]]["image"]
        blob = value["bytes"] if isinstance(value, dict) else value
        if hashlib.sha256(blob).hexdigest() != item["image_sha256"]:
            raise ValueError("Selected image changed")
        return Image.open(io.BytesIO(blob)).convert("RGB")
    processor = ColQwen2_5_Processor.from_pretrained(config["processor_model"], local_files_only=True)
    processor.image_processor.max_pixels = protocol["max_pixels"]
    processor.image_processor.size["longest_edge"] = protocol["max_pixels"]
    spec = config["source"] if args.phase == "source" else config["targets"][args.phase]
    model, base, diagnostics = load_model(spec, config, torch)
    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - start
    if args.phase == "source":
        source_bank = []
        with torch.inference_mode():
            for i, item in enumerate(inputs["pages"]):
                batch = processor.process_images([image_of(item)]).to("cuda:0")
                state = capture(base, batch, torch)
                if not torch.isfinite(state["vision"]).all():
                    raise RuntimeError("Nonfinite source state")
                torch.save(state, root / "states" / f"{i:04d}.pt")
                if protocol.get("compare_stale_index", False):
                    source_bank.append(replay(base, state, torch).cpu())
                if i % 20 == 0:
                    print(json.dumps({"phase": "source", "pages": i + 1}), flush=True)
        if source_bank:
            torch.save(source_bank, root / "source-documents.pt")
        write(root / "source.json", {"diagnostics": diagnostics, "pid": os.getpid(),
              "processor": processor.to_dict(),
              "states": [{"file": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                         for p in sorted((root / "states").glob("*.pt"))]})
        return
    source = json.loads((root / "source.json").read_text())
    records, raw_bank, replay_bank = [], [], []
    def raw(item):
        batch = processor.process_images([image_of(item)]).to("cuda:0")
        values = model(**batch)
        return values[0][batch["attention_mask"][0].bool()].detach().cpu()
    def restored(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        return replay(base, state, torch).detach().cpu()
    with torch.inference_mode():
        # Warm both routes; no warmup observations enter the timing rows.
        raw(inputs["pages"][0])
        restored(root / "states" / "0000.pt")
        for i, item in enumerate(inputs["pages"]):
            path = root / "states" / f"{i:04d}.pt"
            if digest(path) != source["states"][i]["sha256"]:
                raise RuntimeError("Stored state changed")
            if i % 2:
                b, tb = timed(lambda: restored(path))
                a, ta = timed(lambda: raw(item))
            else:
                a, ta = timed(lambda: raw(item))
                b, tb = timed(lambda: restored(path))
            shape_equal = a.shape == b.shape
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            rec = {**item, "raw_shape": list(a.shape), "replay_shape": list(b.shape),
                   "finite": finite, "tensor_equal": bool(shape_equal and torch.equal(a, b)),
                   "bit_equal": bool(shape_equal and a.dtype == b.dtype and
                       torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))),
                   "elements": a.numel(), "equal_elements": int((a == b).sum()) if shape_equal else 0,
                   "max_abs_error": float((a.float()-b.float()).abs().max()) if shape_equal else None,
                   "raw_page_seconds": ta, "replay_with_read_seconds": tb}
            records.append(rec)
            raw_bank.append(a)
            replay_bank.append(b)
            with (root / f"{args.phase}-pages.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 20 == 0:
                print(json.dumps({"phase": args.phase, "pages": i + 1, "equal": rec["tensor_equal"],
                                  "max_abs_error": rec["max_abs_error"]}), flush=True)
        queries = []
        texts = [q["text"] for q in inputs["queries"]]
        for start in range(0, len(texts), protocol["query_batch_size"]):
            batch = processor.process_queries(texts[start:start + protocol["query_batch_size"]]).to("cuda:0")
            vectors = model(**batch)
            queries.extend(v[m.bool()].cpu() for v, m in zip(vectors, batch["attention_mask"]))
        # Separate scoring calls; no copying target scores or ranks into replay.
        a = streaming_maxsim(queries, raw_bank, device="cuda:0", query_chunk=8, document_chunk=4)
        b = streaming_maxsim(queries, replay_bank, device="cuda:0", query_chunk=8, document_chunk=4)
    order_a = np.argsort(-a, axis=1, kind="stable")[:, :10]
    order_b = np.argsort(-b, axis=1, kind="stable")[:, :10]
    ordered = (order_a == order_b).all(axis=1)
    if protocol.get("compare_stale_index", False):
        stale_bank = torch.load(root / "source-documents.pt", weights_only=True, map_location="cpu")
        stale_scores = streaming_maxsim(queries, stale_bank, device="cuda:0", query_chunk=8, document_chunk=4)
        stale_order = np.argsort(-stale_scores, axis=1, kind="stable")[:, :10]
        write(root / "stale-rankings.json", {"stale_top10": stale_order.tolist(),
              "target_top10": order_a.tolist(), "replay_top10": order_b.tolist()})
    write(root / f"{args.phase}-rankings.json", {"raw_top10": order_a.tolist(), "replay_top10": order_b.tolist()})
    torch.save({"raw": raw_bank, "replay": replay_bank, "queries": queries}, root / f"{args.phase}-banks.pt")
    write(root / f"{args.phase}-result.json", {
        "target": args.phase, "pid": os.getpid(), "source_pid": source["pid"], "diagnostics": diagnostics,
        "pages": len(records), "tensor_equal_pages": sum(r["tensor_equal"] for r in records),
        "finite_pages": sum(r["finite"] for r in records), "elements": sum(r["elements"] for r in records),
        "equal_elements": sum(r["equal_elements"] for r in records),
        "max_abs_error": max((r["max_abs_error"] for r in records if r["max_abs_error"] is not None), default=None),
        "queries": len(queries), "ranking_gallery_pages": len(records),
        "ordered_top10_equal_queries": int(ordered.sum()), "score_arrays_equal": bool(np.array_equal(a, b)),
        "ta_at_10": float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(order_a, order_b)])),
        "raw_page_seconds": sum(r["raw_page_seconds"] for r in records),
        "replay_with_read_seconds": sum(r["replay_with_read_seconds"] for r in records),
        "timing_scope": protocol["measurements"][2], "physical_index_equality": "not tested"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="orchestrate")
    args = parser.parse_args()
    config, protocol = json.loads(args.config.read_text()), json.loads(args.protocol.read_text())
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES required")
    if args.phase != "orchestrate":
        worker(args, config, protocol)
        return
    if protocol["status"] != "frozen-before-gpu-output":
        raise ValueError("Protocol not frozen")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "states").mkdir()
    # Pin all model shards, configs, processor/tokenizer files and adapter payloads before GPU output.
    artifacts = {}
    roots = [config["base_model"], config["processor_model"], config["source"]["adapter"]]
    roots += [s["adapter"] for s in config["targets"].values()]
    for root in dict.fromkeys(roots):
        for p in sorted(Path(root).iterdir()):
            if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model"):
                artifacts[str(p)] = digest(p)
    for name, spec in {"source": config["source"], **config["targets"]}.items():
        pin = protocol["source"] if name == "source" else protocol["targets"][name]
        if digest(Path(spec["adapter"]) / "adapter_model.safetensors") != pin["adapter_sha256"]:
            raise ValueError(f"Adapter hash mismatch: {name}")
        if digest(spec["projection"]) != pin["projection_sha256"]:
            raise ValueError(f"Projection hash mismatch: {name}")
        artifacts[spec["projection"]] = digest(spec["projection"])
    pages, queries, datasets = select(config, protocol)
    write(args.output / "inputs.json", {"pages": pages, "queries": queries, "dataset_sha256": datasets})
    import torch
    write(args.output / "manifest.json", {"protocol_sha256": digest(args.protocol), "config": config,
          "code_sha256": {str(p): digest(p) for p in [Path(__file__), *Path(__file__).resolve().parents[1].joinpath("support").glob("*.py")]},
          "artifacts_sha256": artifacts, "environment": {n: importlib.metadata.version(n) for n in
          ("torch", "transformers", "peft", "colpali-engine", "numpy", "pyarrow", "safetensors")},
          "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
          "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "shared_gpu": True,
          "cudnn_sdpa": False, "allow_tf32": True})
    write(args.output / "protocol.json", protocol)
    for phase in ["source", *config["targets"]]:
        subprocess.run([sys.executable, __file__, "--config", str(args.config), "--protocol", str(args.protocol),
                        "--output", str(args.output), "--phase", phase], check=True)
    write(args.output / "result.json", {name: json.loads((args.output / f"{name}-result.json").read_text())
                                       for name in config["targets"]})


if __name__ == "__main__":
    main()
