"""Controlled mechanism tests; synthetic interventions are never release evidence."""
import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reprforge.planning import MaterializationOption, UpdateScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from PIL import Image

    script = Path(__file__).parents[1] / "reconstruction/run.py"
    spec = importlib.util.spec_from_file_location("endpoint", script)
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    manifest = json.loads((args.prior / "manifest.json").read_text())
    config = manifest["config"]
    for name, expected in manifest["artifacts_sha256"].items():
        assert ep.digest(name) == expected, name
    all_inputs = json.loads((args.prior / "inputs.json").read_text())["pages"]
    if not 1 <= args.pages <= len(all_inputs):
        raise ValueError("pages must be within the frozen prior sample")
    inputs = all_inputs[:args.pages]
    state_records = json.loads((args.prior / "source.json").read_text())["states"][:args.pages]
    if args.pages != 8:
        protocol = dict(protocol, id="core-mechanism-v2",
            sample=f"First {args.pages} pages from the frozen prior; no outcome selection.",
            scope=f"{args.pages} pages x 5 unchanged interventions; no timing or population claim.")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    proc = ColQwen2_5_Processor.from_pretrained(config["processor_model"], local_files_only=True)
    proc.image_processor.max_pixels = 12845056
    proc.image_processor.size["longest_edge"] = 12845056
    model, base, diagnostics = ep.load_model(config["targets"]["vidore-v0.2"], config, torch)
    rows = {n: pq.read_table(s["parquet"], columns=["image"]).to_pylist()
            for n, s in config["collections"].items()}
    blobs, states, stale_text = [], [], []
    def tensor_digest(t):
        return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    with torch.inference_mode():
        for item, rec in zip(inputs, state_records, strict=True):
            value = rows[item["collection"]][item["row"]]["image"]
            blob = value["bytes"] if isinstance(value, dict) else value
            assert hashlib.sha256(blob).hexdigest() == item["image_sha256"]
            path = args.prior / "states" / rec["file"]
            assert ep.digest(path) == rec["sha256"]
            state = torch.load(path, weights_only=True, map_location="cpu")
            blobs.append(blob)
            states.append(state)
            stale_text.append(base.get_input_embeddings()(state["input_ids"].cuda()).cpu())
    token_ids = torch.unique(torch.cat([s["input_ids"].flatten() for s in states])).cuda()
    token_ids = token_ids[token_ids != base.config.image_token_id]
    embedding = base.get_input_embeddings().weight
    patch = base.visual.patch_embed.proj.weight
    merger = base.visual.merger.mlp[2].bias
    saved = {"embedding": embedding[token_ids].detach().clone(),
             "vision": patch.detach().clone(), "merger": merger.detach().clone()}
    factorized = MaterializationOption("visual_tokens_and_context", frozenset({"source", "processor", "vision"}), 0, 0)
    fused = MaterializationOption("fused_language_input", factorized.depends_on | {"base_embedding"}, 0, 0)
    calls = [0]
    def count_vision(_module, _inputs, _output):
        calls[0] += 1
    counter = base.visual.register_forward_hook(count_vision)
    def compare(a, b):
        shape = a.shape == b.shape
        return {"shape_equal": shape,
                "bit_equal": shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                "max_abs_error": float((a.float() - b.float()).abs().max()) if shape else None}
    result, tensor_banks, interventions = [], [], {}
    with torch.inference_mode():
        for case in protocol["cases"]:
            embedding[token_ids] = saved["embedding"]
            patch.copy_(saved["vision"])
            merger.copy_(saved["merger"])
            assert tensor_digest(embedding[token_ids]) == tensor_digest(saved["embedding"])
            assert tensor_digest(patch) == tensor_digest(saved["vision"])
            assert tensor_digest(merger) == tensor_digest(saved["merger"])
            changed = {"adapter"}
            if case == "target_text_embedding":
                value = saved["embedding"].clone()
                value[:, ::2] += 0.125
                embedding[token_ids] = value
                changed.add("base_embedding")
            elif case == "target_vision":
                patch.mul_(1.01)
                changed.add("vision")
            elif case == "target_merger":
                merger[::2] += 0.125
                changed.add("vision")
            interventions[case] = {
                "changed_components": sorted(changed),
                "embedding_before": tensor_digest(saved["embedding"]), "embedding_after": tensor_digest(embedding[token_ids]),
                "vision_before": tensor_digest(saved["vision"]), "vision_after": tensor_digest(patch),
                "merger_before": tensor_digest(saved["merger"]), "merger_after": tensor_digest(merger),
            }
            scenario = UpdateScenario(case, frozenset(changed))
            for index, (blob, state) in enumerate(zip(blobs, states, strict=True)):
                def native():
                    batch = proc.process_images([Image.open(io.BytesIO(blob)).convert("RGB")]).to("cuda:0")
                    return model(**batch)[0][batch["attention_mask"][0].bool()].cpu()
                calls[0] = 0
                raw = native()
                raw_calls = calls[0]
                calls[0] = 0
                replay = ep.replay(base, state, torch).cpu()
                replay_calls = calls[0]
                checks = {"factorized_replay": compare(raw, replay)}
                extra = None
                if case in ("release_control", "target_text_embedding"):
                    def old_text(_module, _inputs, output):
                        return stale_text[index].to(output)
                    hook = base.get_input_embeddings().register_forward_hook(old_text)
                    try:
                        extra = ep.replay(base, state, torch).cpu()
                    finally:
                        hook.remove()
                    checks["stale_fused_replay"] = compare(raw, extra)
                elif case in ("target_vision", "target_merger"):
                    extra = native()
                    checks["raw_fallback"] = compare(raw, extra)
                elif case == "missing_positions":
                    original_rope = base.get_rope_index
                    def zero_positions(*a, **kw):
                        pos, delta = original_rope(*a, **kw)
                        return torch.zeros_like(pos), delta
                    base.get_rope_index = zero_positions
                    try:
                        extra = ep.replay(base, state, torch).cpu()
                    finally:
                        base.get_rope_index = original_rope
                    checks["zero_position_replay"] = compare(raw, extra)
                rec = {"case": case, "index": index, "page": inputs[index],
                       "state_sha256": state_records[index]["sha256"],
                       "factorized_admitted": factorized.remains_valid(scenario),
                       "fused_admitted": fused.remains_valid(scenario),
                       "raw_vision_calls": raw_calls, "replay_vision_calls": replay_calls,
                       "checks": checks}
                result.append(rec)
                tensor_banks.append({"case": case, "index": index, "raw": raw, "replay": replay, "extra": extra})
                with (args.output / "pages.jsonl").open("a") as out:
                    out.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
    counter.remove()
    torch.save(tensor_banks, args.output / "banks.pt")
    summary = {"protocol": protocol, "diagnostics": diagnostics, "interventions": interventions,
               "by_case": {case: {name: sum(r["checks"][name]["bit_equal"] for r in result if r["case"] == case)
                   for name in next(r["checks"] for r in result if r["case"] == case)} for case in protocol["cases"]},
               "pages_per_case": len(inputs), "rows": len(result),
               "all_native_called_vision": all(r["raw_vision_calls"] > 0 for r in result),
               "all_replay_skipped_vision": all(r["replay_vision_calls"] == 0 for r in result),
               "all_checks_finite": all(c["finite"] for r in result for c in r["checks"].values()),
               "banks_sha256": ep.digest(args.output / "banks.pt"),
               "code_sha256": {str(p.relative_to(Path(__file__).parents[2])): ep.digest(p) for p in
                  (Path(__file__), script, Path(__file__).parents[2] / "reprforge/planning.py")},
               "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "dtype": "BF16", "batch": 1},
               "limitation": "Controlled mechanism interventions, not prevalence or ranking/efficiency evidence. Stale fused text is reconstructed from the original shared base embedding, not a separately emitted historical fused checkpoint."}
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
