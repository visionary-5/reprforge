#!/usr/bin/env python3
"""Trace the consumed visual interface of ColQwen2.5 and check the generic hooked resume against full target encoding."""
import io, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import pyarrow.parquet as pq
from PIL import Image
from colpali_engine.models import ColQwen2_5_Processor
from reprforge.integrations.colqwen import load_model, capture as hand_capture, replay as hand_replay
from reprforge.tracing import discover_interface
from reprforge import hooked

config = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cuda.enable_cudnn_sdp(False)
processor = ColQwen2_5_Processor.from_pretrained(config["processor_model"], local_files_only=True)
processor.image_processor.max_pixels = config["max_pixels"]; processor.image_processor.size["longest_edge"] = config["max_pixels"]
corpus = json.load(open(config["corpus"]))["pages"][:n_pages]
def image_of(page):
    col = pq.ParquetFile(page["parquet"]).read_row_group(page["rg"], columns=[page["image_column"]]).column(page["image_column"]).to_pylist()[page["idx"]]
    blob = col["bytes"] if isinstance(col, dict) else col
    return Image.open(io.BytesIO(blob)).convert("RGB")
images = [image_of(p) for p in corpus]

# --- 1. trace on the source
model, base, _ = load_model(config["source"], config, torch)
batch = processor.process_images([images[0]]).to("cuda:0")
t = time.perf_counter()
with torch.inference_mode():
    report = discover_interface(base, lambda _model=model: _model(**batch), visual=[batch["pixel_values"]], text=[batch["input_ids"]],
                                structural=[batch["image_grid_thw"], batch["attention_mask"]])
trace_seconds = time.perf_counter() - t
hand = sorted("visual." + k for k in base.visual.state_dict().keys())
traced = sorted(report.pre_cut_params)
summary = {"trace_seconds": trace_seconds, "op_count": report.op_count,
           "frontier": [f.to_dict() for f in report.frontier], "cut_modules": report.cut_modules, "visual_modules": report.visual_modules,
           "pre_cut_params": len(traced), "post_cut_params": len(report.post_cut_params), "structural_params": report.structural_params,
           "unused_params": report.unused_params[:20], "unused_count": len(report.unused_params),
           "hand_written_visual_state_dict": len(hand), "traced_equals_hand": traced == hand,
           "traced_minus_hand": sorted(set(traced) - set(hand))[:20], "hand_minus_traced": sorted(set(hand) - set(traced))[:20],
           "contract_digest": report.contract_digest()}
print(json.dumps(summary, indent=1, default=str)[:6000], flush=True)
(out / "interface-report.json").write_text(json.dumps(report.to_dict(), indent=1, default=str))

# --- 2. generic capture on the source for n pages (+ hand-written capture for comparison)
spec = hooked.CutSpec.from_report(report, list(report.execution_order))
states, hand_states, digests_src = [], [], hooked.dependency_digest(base, list(report.pre_cut_params))
with torch.inference_mode():
    for img in images:
        b = processor.process_images([img]).to("cuda:0")
        st = hooked.capture(base, spec, lambda _model=model: _model(**b))
        st.pop("result"); st["inputs"] = {k: v.cpu() for k, v in b.items() if k != "pixel_values"}
        states.append(st)
        hand_states.append(hand_capture(base, b, torch))
del model, base; torch.cuda.empty_cache()

# --- 3. target: model-level check via traced dependency set, then generic resume vs full encoding vs hand-written replay
results = {}
for tname, tspec in config["targets"].items():
    tmodel, tbase, _ = load_model(tspec, config, torch)
    digests_tgt = hooked.dependency_digest(tbase, list(report.pre_cut_params))
    model_level_ok = digests_tgt == digests_src
    rows = []
    with torch.inference_mode():
        for img, st, hs in zip(images, states, hand_states):
            b = processor.process_images([img]).to("cuda:0")
            full = tmodel(**b)[0][b["attention_mask"][0].bool()].cpu()
            generic = hooked.resume(tbase, st, lambda _model=tmodel: _model(**b))[0][b["attention_mask"][0].bool()].cpu()
            hand = hand_replay(tbase, hs, torch).cpu()
            rows.append({"generic_bitwise": bool(full.shape == generic.shape and torch.equal(full.view(torch.uint8), generic.view(torch.uint8))),
                         "hand_bitwise": bool(full.shape == hand.shape and torch.equal(full.view(torch.uint8), hand.view(torch.uint8))),
                         "generic_max_err": float((full.float() - generic.float()).abs().max()) if full.shape == generic.shape else None})
    results[tname] = {"model_level_ok": model_level_ok, "pages": len(rows), "generic_bitwise": sum(r["generic_bitwise"] for r in rows),
                      "hand_bitwise": sum(r["hand_bitwise"] for r in rows), "max_err": max(r["generic_max_err"] or 0 for r in rows)}
    print(tname, json.dumps(results[tname]), flush=True)
    del tmodel, tbase; torch.cuda.empty_cache()
(out / "summary.json").write_text(json.dumps({"trace": summary, "targets": results}, indent=1, default=str))
