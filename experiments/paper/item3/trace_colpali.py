#!/usr/bin/env python3
"""Third architecture: ColPali (PaliGemma: SigLIP vision tower + linear projector + Gemma). Trace the consumed visual
interface with no architecture-specific code, then run the generic capture/resume across released adapters on the same base."""
import io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch, pyarrow.parquet as pq
from PIL import Image
from peft import PeftModel
from colpali_engine.models import ColPali, ColPaliProcessor
from reprforge.tracing import discover_interface
from reprforge import hooked

cfg = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); n_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.manual_seed(0); device = "cuda:0"
corpus = json.load(open(cfg["corpus"]))["pages"][:n_pages]
def image_of(page):
    col = pq.ParquetFile(page["parquet"]).read_row_group(page["rg"], columns=[page["image_column"]]).column(page["image_column"]).to_pylist()[page["idx"]]
    blob = col["bytes"] if isinstance(col, dict) else col
    return Image.open(io.BytesIO(blob)).convert("RGB")
images = [image_of(p) for p in corpus]
processor = ColPaliProcessor.from_pretrained(cfg["processor"], local_files_only=True)
def load(adapter):
    base = ColPali.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    if adapter:
        wrapped = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True).eval()
        model = wrapped.get_base_model()
    else:
        model = base
    return model.to(device).eval()
def vec(o, b): return o[0][b["attention_mask"][0].bool()].detach().cpu()

src = load(cfg["source"])
b0 = {k: v.to(device) for k, v in processor.process_images([images[0]]).items()}
t = time.perf_counter()
with torch.inference_mode():
    report = discover_interface(src, lambda _model=src: _model(**b0), visual=[b0["pixel_values"]], text=[b0["input_ids"]], structural=[b0["attention_mask"]])
summary = {"trace_seconds": time.perf_counter() - t, "op_count": report.op_count, "cut_modules": report.cut_modules, "visual_modules": report.visual_modules,
           "pre_cut_params": len(report.pre_cut_params), "post_cut_params": len(report.post_cut_params), "structural_params": report.structural_params,
           "unused_count": len(report.unused_params), "unused_sample": report.unused_params[:8], "contract_digest": report.contract_digest(),
           "frontier": [f.to_dict() for f in report.frontier], "execution_order_head": report.execution_order[:6]}
print(json.dumps({k: v for k, v in summary.items() if k != "frontier"}, indent=1, default=str), flush=True)
for f in report.frontier: print(json.dumps({k: v for k, v in f.to_dict().items() if k != "param_dependencies"}), flush=True)
(out / "interface-report.json").write_text(json.dumps(report.to_dict(), indent=1, default=str))
spec = hooked.CutSpec.from_report(report, list(report.execution_order)); print("spec", spec, flush=True)
src_digest = hooked.dependency_digest(src, list(report.pre_cut_params))
states = []
with torch.inference_mode():
    for img in images:
        b = {k: v.to(device) for k, v in processor.process_images([img]).items()}
        st = hooked.capture(src, spec, lambda _model=src: _model(**b)); st.pop("result"); states.append(st)
mib = sum(t.numel() * t.element_size() for s in states for m in spec.frontier_modules for t in s["modules"][m]["output"] if t is not None) / len(states) / 2**20
print("captured", len(states), "state MiB/page", mib, flush=True)
del src; torch.cuda.empty_cache()
results = {"state_mib_per_page": mib}
for name, adapter in cfg["targets"].items():
    tgt = load(adapter)
    ok = hooked.dependency_digest(tgt, list(report.pre_cut_params)) == src_digest
    rows = []
    with torch.inference_mode():
        for img, st in zip(images, states):
            b = {k: v.to(device) for k, v in processor.process_images([img]).items()}
            full = vec(tgt(**b), b)
            try:
                gen = vec(hooked.resume(tgt, st, lambda _model=tgt: _model(**b)), b)
                rows.append({"bitwise": bool(full.shape == gen.shape and torch.equal(full.view(torch.uint8), gen.view(torch.uint8))), "max_err": float((full.float() - gen.float()).abs().max())})
            except hooked.PageLevelMismatch as e:
                rows.append({"bitwise": False, "page_level_reject": str(e)})
    results[name] = {"model_level_ok": ok, "pages": len(rows), "generic_bitwise": sum(r["bitwise"] for r in rows), "max_err": max((r.get("max_err") or 0) for r in rows),
                     "page_level_rejects": sum("page_level_reject" in r for r in rows)}
    print(name, json.dumps(results[name]), flush=True)
    del tgt; torch.cuda.empty_cache()
(out / "summary.json").write_text(json.dumps({"trace": summary, "targets": results}, indent=1, default=str))
