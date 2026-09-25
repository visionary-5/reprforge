#!/usr/bin/env python3
"""Trace the consumed visual interface of a Qwen3-VL retriever (SauerkrautLM-ColQwen3 2B) and check the generic
hooked resume against full target encoding for the RuDoRe adapter transition. Runs in the qwen3 probe venv."""
import io, json, sys, time
import os
from pathlib import Path
PROBE = Path(os.path.expandvars("${INPUTS}/qwen3-probe"))
sys.path.insert(0, str(PROBE / "scripts")); sys.path.insert(0, str(PROBE / "rudore_probe" / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch, pyarrow.parquet as pq
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2VLImageProcessor
import run_rudore_probe as rr
from producer_modeling import ColQwen3
from reprforge.tracing import discover_interface
from reprforge import hooked

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); n_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 8
device = torch.device("cuda:0"); torch.manual_seed(0)
source_load = PROBE / "rudore_probe" / "source_load"
processor = rr.ExplicitProcessor(image_processor=Qwen2VLImageProcessor.from_pretrained(source_load), tokenizer=AutoTokenizer.from_pretrained(source_load))
data = Path(os.path.expandvars("${INPUTS}/vidore-official-a70f23a/data/vidore_v3_hr/corpus/test-00000-of-00001.parquet"))
ids = sorted(pq.read_table(data, columns=["corpus_id"]).column(0).to_pylist())[:n_pages]
rows = sorted(pq.read_table(data, columns=["corpus_id", "image"], filters=[("corpus_id", "in", ids)]).to_pylist(), key=lambda r: r["corpus_id"])
def batch_of(row):
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    return {k: v.to(device) for k, v in rr.process(processor, img).items()}
def vectors(out_, b):
    return out_[0][b["attention_mask"][0].bool()].detach().cpu()

model = ColQwen3.from_pretrained(source_load, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(device)
b0 = batch_of(rows[0])
t = time.perf_counter()
with torch.inference_mode():
    report = discover_interface(model, lambda _model=model: _model(**b0), visual=[b0["pixel_values"]], text=[b0["input_ids"]],
                                structural=[b0["image_grid_thw"], b0["attention_mask"], b0["mm_token_type_ids"]])
trace_seconds = time.perf_counter() - t
visual_prefixes = tuple(m + "." for m in report.visual_modules)
hand = sorted(n for n, _ in list(model.named_parameters()) + list(model.named_buffers()) if n.startswith("visual."))
summary = {"trace_seconds": trace_seconds, "op_count": report.op_count, "frontier": [f.to_dict() for f in report.frontier],
           "cut_modules": report.cut_modules, "visual_modules": report.visual_modules, "pre_cut_params": len(report.pre_cut_params),
           "post_cut_params": len(report.post_cut_params), "structural_params": report.structural_params, "unused_count": len(report.unused_params),
           "unused_sample": report.unused_params[:10],
           "deepstack_in_pre_cut": sorted({p.split(".")[1] for p in report.pre_cut_params if "deepstack" in p}),
           "hand_visual_prefix_params": len(hand), "traced_minus_hand": sorted(set(report.pre_cut_params) - set(hand))[:10],
           "hand_minus_traced": sorted(set(hand) - set(report.pre_cut_params))[:10], "contract_digest": report.contract_digest()}
print(json.dumps({k: v for k, v in summary.items() if k != "frontier"}, indent=1, default=str), flush=True)
for f in report.frontier: print(json.dumps({k: v for k, v in f.to_dict().items() if k != "param_dependencies"}), flush=True)
(out / "interface-report.json").write_text(json.dumps(report.to_dict(), indent=1, default=str))

spec = hooked.CutSpec.from_report(report, list(report.execution_order))
src_digest = hooked.dependency_digest(model, list(report.pre_cut_params))
states = []
with torch.inference_mode():
    for row in rows:
        b = batch_of(row); st = hooked.capture(model, spec, lambda _model=model: _model(**b)); st.pop("result"); states.append(st)
state_mib = sum(t.numel() * t.element_size() for s in states for t in s["modules"][spec.frontier_modules[0]]["output"] if t is not None) / len(states) / 2**20
print("captured", len(states), "retained leaves", spec.retained_leaves, "state MiB/page", state_mib, flush=True)

# target: apply the published adapter (no merge), then model-level check on traced dependency set and generic resume
wrapped = PeftModel.from_pretrained(model, PROBE / "rudore_probe" / "adapter", is_trainable=False, autocast_adapter_dtype=False).eval()
target = wrapped.get_base_model()
tgt_digest = hooked.dependency_digest(target, list(report.pre_cut_params))
res = []
with torch.inference_mode():
    for row, st in zip(rows, states):
        b = batch_of(row)
        full = vectors(target(**b), b)
        gen = vectors(hooked.resume(target, st, lambda: target(**b)), b)
        res.append({"bitwise": bool(full.shape == gen.shape and torch.equal(full.view(torch.uint8), gen.view(torch.uint8))),
                    "max_err": float((full.float() - gen.float()).abs().max()) if full.shape == gen.shape else None})
result = {"model_level_ok": src_digest == tgt_digest, "pages": len(res), "generic_bitwise": sum(r["bitwise"] for r in res), "max_err": max(r["max_err"] or 0 for r in res)}
print("RuDoRe", json.dumps(result), flush=True)
# negative control through the same generic path: drop every frontier tensor except the first (main visual tokens)
partial = []
with torch.inference_mode():
    for row, st in zip(rows[:3], states[:3]):
        b = batch_of(row)
        full = vectors(target(**b), b)
        import copy
        st2 = copy.deepcopy(st); rec = st2["modules"][spec.frontier_modules[0]]
        outs = rec["output"]; kept = [i for i, x in enumerate(outs) if x is not None]; rec["output"] = [outs[kept[0]] if i == kept[0] else (torch.zeros_like(x) if x is not None else None) for i, x in enumerate(outs)]
        gen = vectors(hooked.resume(target, st2, lambda: target(**b)), b)
        partial.append(bool(full.shape == gen.shape and torch.equal(full.view(torch.uint8), gen.view(torch.uint8))))
result["state_mib_per_page"] = state_mib; result["control_main_tokens_only_bitwise"] = sum(partial); result["control_pages"] = len(partial)
(out / "summary.json").write_text(json.dumps({"trace": summary, "rudore": result}, indent=1, default=str))
print(json.dumps(result), flush=True)
