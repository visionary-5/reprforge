#!/usr/bin/env python3
"""Qwen3-VL 4B pair with two different publisher wrappers (Tomoro source -> OpenSearch target).
Trace both, match the visual branch across wrappers by traced role, and run the generic resume natively."""
import io, json, sys, time
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoConfig, AutoProcessor, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from reprforge.tracing import discover_interface
from reprforge import hooked

ROOT = Path(os.path.expandvars("${INPUTS}/qwen3/qwen3-4b-pair"))
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); n_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.manual_seed(0); torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

def load(source):
    d = ROOT / ("TomoroAI__tomoro-colqwen3-embed-4b" if source else "OpenSearch-AI__Ops-Colqwen3-4B")
    processor = AutoProcessor.from_pretrained(d, trust_remote_code=True, local_files_only=True)
    clsname = "modeling_colqwen3.ColQwen3" if source else "modeling_ops_colqwen3.OpsColQwen3Model"
    cls = get_class_from_dynamic_module(clsname, str(d), local_files_only=True)
    config = AutoConfig.from_pretrained(d, trust_remote_code=True, local_files_only=True)
    model, info = PreTrainedModel.from_pretrained.__func__(cls, d, config=config, key_mapping=cls._checkpoint_conversion_mapping, local_files_only=True,
                                                            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    if source:
        def forbid(m, a): raise RuntimeError("Uninitialized generation head executed")
        model.vlm.lm_head.register_forward_pre_hook(forbid)
    else:
        model.dims = config.dims
    model.eval().cuda()
    backbone = model.vlm.model if source else model.qwen3vl
    if not source:
        def hook(m, a, kw):
            if kw.get("image_grid_thw") is not None and kw.get("mm_token_type_ids") is None:
                kw["mm_token_type_ids"] = (kw["input_ids"] == model.config.image_token_id).to(torch.int32)
            return a, kw
        backbone.register_forward_pre_hook(hook, with_kwargs=True)
    return model, processor, backbone

data = Path(os.path.expandvars("${INPUTS}/vidore-official-a70f23a/data/vidore_v3_hr/corpus/test-00000-of-00001.parquet"))
ids = sorted(pq.read_table(data, columns=["corpus_id"]).column(0).to_pylist())[:n_pages]
rows = sorted(pq.read_table(data, columns=["corpus_id", "image"], filters=[("corpus_id", "in", ids)]).to_pylist(), key=lambda r: r["corpus_id"])
def batch_of(processor, row):
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    return {k: v.cuda() for k, v in dict(processor.process_images([img])).items()}
def output(model, source, b):
    v = model(**b)
    v = v.embeddings if source else v
    return v[0][b["attention_mask"][0].bool()].detach().cpu()

def trace(model, b):
    structural = [b[k] for k in ("image_grid_thw", "attention_mask", "mm_token_type_ids") if k in b]
    with torch.inference_mode():
        return discover_interface(model, lambda _model=model: _model(**b), visual=[b["pixel_values"]], text=[b["input_ids"]], structural=structural)

smodel, sproc, sback = load(True)
b0 = batch_of(sproc, rows[0])
t = time.perf_counter(); sreport = trace(smodel, b0); ts = time.perf_counter() - t
def brief(r): return {"cut_modules": r.cut_modules, "visual_modules": r.visual_modules, "pre_cut": len(r.pre_cut_params), "post_cut": len(r.post_cut_params),
                      "frontier": [(f.shape, f.source_module, f.source_leaf_index, f.consumer_module, f.consumer_op) for f in r.frontier], "structural": r.structural_params}
print("SOURCE", ts, json.dumps(brief(sreport), default=str), flush=True)
sspec = hooked.CutSpec.from_report(sreport, list(sreport.execution_order))
def relative(digests, module): return {k[len(module) + 1:] if k.startswith(module + ".") else k: v for k, v in digests.items()}
sdig = relative(hooked.dependency_digest(smodel, list(sreport.pre_cut_params)), sspec.cut_modules[0])
states = []
with torch.inference_mode():
    for row in rows:
        b = batch_of(sproc, row); st = hooked.capture(smodel, sspec, lambda _model=smodel: _model(**b)); st.pop("result"); states.append(st)
mib = sum(t.numel() * t.element_size() for s in states for m in sspec.frontier_modules for t in s["modules"][m]["output"] if t is not None) / len(states) / 2**20
print("captured", len(states), "MiB/page", mib, flush=True)
del smodel; torch.cuda.empty_cache()

tmodel, tproc, tback = load(False)
b1 = batch_of(tproc, rows[0])
t = time.perf_counter(); treport = trace(tmodel, b1); tt = time.perf_counter() - t
print("TARGET", tt, json.dumps(brief(treport), default=str), flush=True)
tspec = hooked.CutSpec.from_report(treport, list(treport.execution_order))
assert len(tspec.cut_modules) == len(sspec.cut_modules) == 1
mapping = dict(zip(sspec.cut_modules, tspec.cut_modules))
tdig = relative(hooked.dependency_digest(tmodel, list(treport.pre_cut_params)), tspec.cut_modules[0])
model_level_ok = sdig == tdig
print("model-level (relative names):", model_level_ok, len(sdig), len(tdig), "interface digests equal after prefix strip:",
      sorted(k[len(sspec.cut_modules[0])+1:] for k in sreport.pre_cut_params) == sorted(k[len(tspec.cut_modules[0])+1:] for k in treport.pre_cut_params), flush=True)
res = []
with torch.inference_mode():
    for row, st in zip(rows, states):
        b = batch_of(tproc, row)
        full = output(tmodel, False, b)
        st2 = {"spec": st["spec"], "modules": {mapping[k]: v for k, v in st["modules"].items()}}
        try:
            gen = hooked.resume(tmodel, st2, lambda _model=tmodel: _model(**b))[0][b["attention_mask"][0].bool()].detach().cpu()
            res.append({"bitwise": bool(full.shape == gen.shape and torch.equal(full.view(torch.uint8), gen.view(torch.uint8))), "max_err": float((full.float() - gen.float()).abs().max())})
        except hooked.PageLevelMismatch as e:
            res.append({"bitwise": False, "page_level_reject": str(e)})
result = {"model_level_ok": model_level_ok, "pages": len(res), "generic_bitwise": sum(r["bitwise"] for r in res), "page_level_rejects": sum("page_level_reject" in r for r in res),
          "max_err": max((r.get("max_err") or 0) for r in res), "state_mib_per_page": mib, "module_mapping": mapping, "trace_seconds": [ts, tt]}
print("RESULT", json.dumps(result), flush=True)
(out / "summary.json").write_text(json.dumps({"source": brief(sreport), "target": brief(treport), "result": result}, indent=1, default=str))
(out / "source-interface.json").write_text(json.dumps(sreport.to_dict(), indent=1, default=str)); (out / "target-interface.json").write_text(json.dumps(treport.to_dict(), indent=1, default=str))
