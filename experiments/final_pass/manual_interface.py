"""Matched expert and traced contracts; shared capture, storage and resume."""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from reprforge import hooked
from reprforge.tracing import discover_interface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["qwen25", "qwen3"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pages", type=int, default=8)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    if args.architecture == "qwen25":
        from colpali_engine.models import ColQwen2_5_Processor
        from reprforge.integrations.colqwen import load_model

        processor = ColQwen2_5_Processor.from_pretrained(
            cfg["processor_model"], local_files_only=True
        )
        processor.image_processor.max_pixels = 602112
        processor.image_processor.size["longest_edge"] = 602112
        model, base, _ = load_model(cfg["source"], cfg, torch)
        process = lambda img: processor.process_images([img])
        manual = hooked.CutSpec(("visual",), ("visual",), {"visual": (0,)})
    else:
        import qwen3_loader

        qwen3_loader.ROOT = Path(cfg["qwen3_root"])
        model, processor, _ = qwen3_loader.load(True)
        base = model
        process = lambda img: processor.process_images([img])
        # Expert contract: main merged output plus all three DeepStack outputs.
        # Qwen3's ModelOutput has an unused last_hidden_state at leaf 0.
        manual = hooked.CutSpec(
            ("vlm.model.visual",),
            ("vlm.model.visual",),
            {"vlm.model.visual": (1, 2, 3, 4)},
        )
    rows = sorted(
        pq.read_table(cfg["hr_corpus"]).to_pylist(), key=lambda r: r["corpus_id"]
    )[: args.pages]
    manifest = [
        {
            "id": r["corpus_id"],
            "sha256": hashlib.sha256(r["image"]["bytes"]).hexdigest(),
        }
        for r in rows
    ]
    (out / "pages.json").write_text(json.dumps(manifest, indent=2))

    def batch(row):
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return {k: v.cuda() for k, v in process(image).items()}

    b = batch(rows[0])
    structural = [v for k, v in b.items() if k not in ("pixel_values", "input_ids")]
    with torch.inference_mode():
        model(**b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        report = discover_interface(
            base,
            lambda: model(**b),
            visual=[b["pixel_values"]],
            text=[b["input_ids"]],
            structural=structural,
        )
        torch.cuda.synchronize()
        trace_seconds = time.perf_counter() - start
        auto = hooked.CutSpec.from_report(report, list(report.execution_order))
        # Independent expert rule, including non-persistent registered buffers.
        prefix = manual.cut_modules[0] + "."
        manual_deps = sorted(
            n
            for n, _ in list(base.named_parameters()) + list(base.named_buffers())
            if n.startswith(prefix)
        )
        auto_deps = sorted(report.pre_cut_params)
        assert manual_deps == auto_deps
        assert manual == auto, (manual, auto)
        src = {
            n[len(prefix) :]: d
            for n, d in hooked.dependency_digest(base, manual_deps).items()
        }
        for i, row in enumerate(rows):
            b = batch(row)
            for label, spec in [("manual", manual), ("auto", auto)]:
                st = hooked.capture(base, spec, lambda: model(**b))
                st.pop("result")
                torch.save(st, out / f"{label}-{i}.pt")
    (out / "interface.json").write_text(json.dumps(report.to_dict(), indent=2))
    if args.architecture == "qwen25":
        del model, base
        torch.cuda.empty_cache()
        model, base, _ = load_model(cfg["targets"]["vidore-v0.2"], cfg, torch)
    else:
        del model, base
        torch.cuda.empty_cache()
        model, processor, _ = qwen3_loader.load(False)
        base = model
        process = lambda img: processor.process_images([img])
        for i in range(len(rows)):
            for label in ["manual", "auto"]:
                path = out / f"{label}-{i}.pt"
                st = torch.load(path, weights_only=False)
                st["modules"] = {"qwen3vl.visual": st["modules"]["vlm.model.visual"]}
                torch.save(st, path)
    target_prefix = "visual." if args.architecture == "qwen25" else "qwen3vl.visual."
    target_deps = [target_prefix + n for n in src]
    assert src == {
        n[len(target_prefix) :]: d
        for n, d in hooked.dependency_digest(base, target_deps).items()
    }
    b = batch(rows[0])
    with torch.inference_mode():
        target_report = discover_interface(
            base,
            lambda: model(**b),
            visual=[b["pixel_values"]],
            text=[b["input_ids"]],
            structural=[
                v for k, v in b.items() if k not in ("pixel_values", "input_ids")
            ],
        )
    assert sorted(target_report.pre_cut_params) == sorted(target_deps)
    (out / "target-interface.json").write_text(
        json.dumps(target_report.to_dict(), indent=2)
    )
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            b = batch(row)
            full = model(**b).detach().cpu()
            rec = {"id": row["corpus_id"]}
            for label in ["manual", "auto"]:
                state = torch.load(out / f"{label}-{i}.pt", weights_only=False)
                replay = hooked.resume(base, state, lambda: model(**b)).detach().cpu()
                rec[label + "_exact"] = bool(
                    torch.equal(full.view(torch.uint8), replay.view(torch.uint8))
                )
                rec[label + "_max_error"] = float(
                    (full.float() - replay.float()).abs().max()
                )
                assert rec[label + "_exact"]
            results.append(rec)
    result = {
        "architecture": args.architecture,
        "trace_seconds": trace_seconds,
        "dependencies": manual_deps,
        "manual_spec": dataclasses.asdict(manual),
        "auto_spec": dataclasses.asdict(auto),
        "interface_equal": manual == auto,
        "dependencies_equal": manual_deps == auto_deps,
        "rows": results,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"done": len(results), "trace_seconds": trace_seconds}), flush=True
    )


if __name__ == "__main__":
    main()
