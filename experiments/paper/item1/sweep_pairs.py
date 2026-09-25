#!/usr/bin/env python3
"""Upgrade-lifecycle validation sweep: one retained source, many released targets.

For a model family (shared base architecture) the script retains the source's traced visual-branch state on K
pages under two processing configurations (the source's native processor and a common configuration), then, for
every released target in the family:

  1. loads the target as published (adapter on the declared base, or merged weights),
  2. traces the target's consumed visual interface and compares the dependency set with the source's
     (model-level check: same parameter/buffer names below the cut and identical bytes as loaded),
  3. runs the page-level check natively (the digest of what the target's own processor hands to the visual branch
     against what produced the retained state) and under the common configuration,
  4. resumes the target from the retained state wherever both checks pass and compares bitwise with full target
     encoding; where the model-level check fails, runs a forced-reuse control on a few pages so the error of an
     unchecked reuse is measured rather than assumed.

Every target receives a verdict with a reason code. Nothing is skipped because it "looks" incompatible.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(os.environ.get("REPRFORGE_REPO", str(Path(__file__).resolve().parents[3])))
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from reprforge import hooked  # noqa: E402
from reprforge.tracing import discover_interface  # noqa: E402

DEVICE = "cuda:0"


def jdump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, default=str) + "\n")


def image_of(page):
    col = pq.ParquetFile(page["parquet"]).read_row_group(page["rg"], columns=[page["image_column"]]).column(page["image_column"]).to_pylist()[page["idx"]]
    blob = col["bytes"] if isinstance(col, dict) else col
    return Image.open(io.BytesIO(blob)).convert("RGB")


# ----------------------------------------------------------------------------- loaders

class Family:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = cfg["loader"]

    def processor_class(self):
        from colpali_engine import models as m
        return {"colqwen25": m.ColQwen2_5_Processor, "colqwen2": m.ColQwen2Processor, "colpali": m.ColPaliProcessor, "colidefics3": m.ColIdefics3Processor}[self.loader]

    def model_class(self):
        from colpali_engine import models as m
        return {"colqwen25": m.ColQwen2_5, "colqwen2": m.ColQwen2, "colpali": m.ColPali, "colidefics3": m.ColIdefics3}[self.loader]

    def processor(self, path, *, max_pixels=None):
        try:
            p = self.processor_class().from_pretrained(path, local_files_only=True)
        except Exception:  # noqa: BLE001
            # release ships only an image-processor config (no tokenizer files): assemble the release's image
            # processing with the family tokenizer, which is what its own engine version does at load time
            import copy
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            p = copy.deepcopy(self.processor_class().from_pretrained(self.cfg["processor"], local_files_only=True))
            p.image_processor = image_processor
            p.assembled_from_image_processor_only = True
        if max_pixels is not None and hasattr(p, "image_processor") and hasattr(p.image_processor, "max_pixels"):
            p.image_processor.max_pixels = max_pixels
            if isinstance(getattr(p.image_processor, "size", None), dict) and "longest_edge" in p.image_processor.size:
                p.image_processor.size["longest_edge"] = max_pixels
        return p

    def native_processor_path(self, release):
        d = Path(release["path"])
        if (d / "preprocessor_config.json").exists():
            return str(d)
        return release.get("processor") or release.get("base") or self.cfg["processor"]

    def load(self, release):
        kind = release["kind"]
        d = Path(release["path"])
        if kind == "adapter" and not (d / "adapter_config.json").exists() and (list(d.glob("*.safetensors")) or list(d.glob("*.bin"))):
            kind = "merged"  # published as full weights although the census classified it as an adapter
            release["kind_loaded"] = "merged"
        base_path = release.get("base") or self.cfg["base"]  # a target may declare a different base than the family default
        if self.loader == "colqwen25" and kind == "adapter":
            from reprforge.integrations.colqwen import load_model
            spec = {"adapter": release["path"], "projection": release.get("projection") or self.cfg["projection"]}
            model, base, diag = load_model(spec, {"base_model": base_path}, torch)
            return model, base, diag
        cls = self.model_class()
        if kind == "adapter":
            from peft import PeftModel
            base = cls.from_pretrained(base_path, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
            wrapped = PeftModel.from_pretrained(base, release["path"], is_trainable=False, local_files_only=True).eval()
            model = wrapped.get_base_model().to(DEVICE).eval()
            return wrapped.to(DEVICE).eval(), model, {"peft": True}
        model = cls.from_pretrained(release["path"], torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).to(DEVICE).eval()
        return model, model, {"merged": True}


def forward_vectors(model, batch):
    out = model(**batch)
    return out[0][batch["attention_mask"][0].bool()].detach().cpu()


def seeds_of(batch):
    structural = [batch[k] for k in ("image_grid_thw", "attention_mask", "mm_token_type_ids") if k in batch]
    return {"visual": [batch["pixel_values"]], "text": [batch["input_ids"]], "structural": structural}


def trace(model, batch):
    s = seeds_of(batch)
    with torch.inference_mode():
        return discover_interface(model, lambda _model=model: _model(**batch), visual=s["visual"], text=s["text"], structural=s["structural"])


def relative(names, prefix):
    return sorted(n[len(prefix) + 1:] if n.startswith(prefix + ".") else n for n in names)


def category(name):
    parts = name.split(".")
    keep = []
    for p in parts:
        if p.isdigit():
            break
        keep.append(p)
    return ".".join(keep[:3]) or name


def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def bitwise(a, b):
    return bool(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--control-pages", type=int, default=3)
    ap.add_argument("--targets", help="comma-separated repo ids to restrict to")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    allcfg = json.loads(args.config.read_text())
    cfg = allcfg["families"][args.family]
    fam = Family(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    all_pages = json.loads(Path(allcfg["corpus"]).read_text())["pages"]
    # evenly spaced across the interleaved corpus so that every collection (and page size) is represented
    stride = max(1, len(all_pages) // args.pages)
    corpus = [all_pages[i * stride] for i in range(args.pages)]
    images = [image_of(p) for p in corpus]
    jdump(args.out / "pages.json", [{"i": i * stride, "collection": p["collection"], "sha": p["sha"], "bytes": p["bytes"]} for i, p in enumerate(corpus)])
    common_px = cfg.get("common_max_pixels")

    # ---- source: trace + retain under native and common processing
    src_rel = cfg["source"]
    src_model, src_base, src_diag = fam.load(src_rel)
    src_native_proc = fam.processor(fam.native_processor_path(src_rel))
    src_common_proc = fam.processor(cfg["processor"], max_pixels=common_px)
    b0 = to_device(src_common_proc.process_images([images[0]]))
    t = time.perf_counter()
    sreport = trace(src_base, b0)
    trace_seconds = time.perf_counter() - t
    sspec = hooked.CutSpec.from_report(sreport, list(sreport.execution_order))
    cut = sspec.cut_modules[0]
    src_names = relative(sreport.pre_cut_params, cut)
    src_digest = {n[len(cut) + 1:] if n.startswith(cut + ".") else n: d for n, d in hooked.dependency_digest(src_base, list(sreport.pre_cut_params)).items()}
    states = {"native": [], "common": []}
    src_vectors = {"native": [], "common": []}
    with torch.inference_mode():
        for img in images:
            for mode, proc in (("native", src_native_proc), ("common", src_common_proc)):
                b = to_device(proc.process_images([img]))
                st = hooked.capture(src_base, sspec, lambda _model=src_model: _model(**b))
                src_vectors[mode].append(st.pop("result")[0][b["attention_mask"][0].bool()].detach().cpu())
                states[mode].append(st)
    state_mib = sum(t.numel() * t.element_size() for s in states["common"] for m in sspec.frontier_modules for t in s["modules"][m]["output"] if t is not None) / len(images) / 2**20
    source_info = {"repo": src_rel["repo"], "path": src_rel["path"], "kind": src_rel["kind"], "cut_modules": sspec.cut_modules, "frontier_modules": sspec.frontier_modules,
                   "frontier": [f.to_dict() for f in sreport.frontier], "pre_cut_params": len(sreport.pre_cut_params), "post_cut_params": len(sreport.post_cut_params),
                   "structural_params": sreport.structural_params, "trace_seconds": trace_seconds, "state_mib_per_page_common": state_mib,
                   "native_processor": fam.native_processor_path(src_rel), "common_max_pixels": common_px, "pages": len(images),
                   "native_equals_common_inputs": sum(states["native"][i]["modules"][sspec.cut_modules[0]]["input_digest"] == states["common"][i]["modules"][sspec.cut_modules[0]]["input_digest"] for i in range(len(images))),
                   "diagnostics": {k: v for k, v in src_diag.items() if k != "unexpected"}}
    jdump(args.out / "source.json", source_info)
    jdump(args.out / "source-interface.json", sreport.to_dict())
    print(json.dumps({k: v for k, v in source_info.items() if k not in ("frontier", "diagnostics")}, default=str), flush=True)
    del src_model, src_base
    torch.cuda.empty_cache()

    # ---- targets
    rows_path = args.out / "rows.jsonl"
    done = set()
    if args.skip_done and rows_path.exists():
        done = {json.loads(l)["target"] for l in rows_path.read_text().splitlines() if l.strip() and json.loads(l).get("verdict") not in ("LOAD_ERROR", "RUN_ERROR")}
    restrict = set(args.targets.split(",")) if args.targets else None
    for rel in cfg["targets"]:
        if restrict and rel["repo"] not in restrict:
            continue
        if rel["repo"] in done:
            continue
        row = {"source": src_rel["repo"], "target": rel["repo"], "kind": rel["kind"], "family": args.family, "pages": len(images)}
        t0 = time.perf_counter()
        try:
            tmodel, tbase, tdiag = fam.load(rel)
            row["kind_loaded"] = rel.get("kind_loaded", rel["kind"])
        except Exception as e:  # noqa: BLE001
            row.update({"verdict": "LOAD_ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}", "seconds": time.perf_counter() - t0})
            (args.out / f"load-error-{rel['repo'].replace('/', '__')}.txt").write_text(traceback.format_exc())
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(json.dumps(row, default=str), flush=True)
            torch.cuda.empty_cache()
            continue
        try:
            # processors
            native_path = fam.native_processor_path(rel)
            try:
                t_native_proc = fam.processor(native_path)
                processor_note = None
            except Exception as e:  # noqa: BLE001
                t_native_proc = fam.processor(cfg["processor"])
                processor_note = f"native processor unusable ({type(e).__name__}); base processor used"
            t_common_proc = fam.processor(cfg["processor"], max_pixels=common_px)
            # model-level check via traced dependency set
            b0 = to_device(t_common_proc.process_images([images[0]]))
            treport = trace(tbase, b0)
            tspec = hooked.CutSpec.from_report(treport, list(treport.execution_order))
            tcut = tspec.cut_modules[0] if tspec.cut_modules else None
            tgt_digest = {n[len(tcut) + 1:] if tcut and n.startswith(tcut + ".") else n: d for n, d in hooked.dependency_digest(tbase, list(treport.pre_cut_params)).items()}
            added = sorted(set(tgt_digest) - set(src_digest))
            removed = sorted(set(src_digest) - set(tgt_digest))
            changed = sorted(n for n in set(src_digest) & set(tgt_digest) if src_digest[n] != tgt_digest[n])
            model_level_ok = not added and not removed and not changed
            cats = {}
            for n in added + removed + changed:
                cats[category(n)] = cats.get(category(n), 0) + 1
            row["model_level"] = {"ok": model_level_ok, "cut_module": tcut, "frontier_count": len(treport.frontier), "params_compared": len(src_digest),
                                  "added": len(added), "removed": len(removed), "changed": len(changed), "categories": cats,
                                  "examples": (added[:3] + removed[:3] + changed[:3]), "lora_on_visual": any("lora" in n for n in added),
                                  "target_frontier_shapes": [list(f.shape) for f in treport.frontier], "source_frontier_shapes": [list(f.shape) for f in sreport.frontier]}
            mapping = {sspec.cut_modules[0]: tcut} if tcut else {}
            row["processor"] = {"native_path": native_path, "note": processor_note, "assembled": getattr(t_native_proc, "assembled_from_image_processor_only", False),
                                "native_image_processor": getattr(getattr(t_native_proc, "image_processor", None), "to_dict", lambda: {})()}
            # page-level native audit + dynamic checks
            native_pages = {"accepted": 0, "rejected": 0, "bitwise": 0, "max_err": 0.0}
            common_pages = {"accepted": 0, "rejected": 0, "bitwise": 0, "max_err": 0.0}
            forced = None
            with torch.inference_mode():
                if model_level_ok:
                    for mode, proc, res in (("native", t_native_proc, native_pages), ("common", t_common_proc, common_pages)):
                        for img, st in zip(images, states[mode]):
                            b = to_device(proc.process_images([img]))
                            full = forward_vectors(tmodel, b)
                            st2 = {"spec": st["spec"], "modules": {mapping.get(k, k): v for k, v in st["modules"].items()}}
                            try:
                                gen = hooked.resume(tbase, st2, lambda _model=tmodel: _model(**b))[0][b["attention_mask"][0].bool()].detach().cpu()
                            except hooked.PageLevelMismatch:
                                res["rejected"] += 1
                                continue
                            res["accepted"] += 1
                            res["bitwise"] += int(bitwise(full, gen))
                            if full.shape == gen.shape:
                                res["max_err"] = max(res["max_err"], float((full.float() - gen.float()).abs().max()))
                else:
                    forced = {"pages": 0, "bitwise": 0, "max_err": 0.0, "page_level_rejected": 0}
                    for img, st in zip(images[: args.control_pages], states["common"][: args.control_pages]):
                        b = to_device(t_common_proc.process_images([img]))
                        full = forward_vectors(tmodel, b)
                        st2 = {"spec": st["spec"], "modules": {mapping.get(k, k): v for k, v in st["modules"].items()}}
                        try:
                            gen = hooked.resume(tbase, st2, lambda _model=tmodel: _model(**b))[0][b["attention_mask"][0].bool()].detach().cpu()
                        except hooked.PageLevelMismatch:
                            forced["page_level_rejected"] += 1
                            continue
                        except Exception as e:  # noqa: BLE001
                            forced["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                            break
                        forced["pages"] += 1
                        forced["bitwise"] += int(bitwise(full, gen))
                        if full.shape == gen.shape:
                            forced["max_err"] = max(forced["max_err"], float((full.float() - gen.float()).abs().max()))
                        else:
                            forced["shape_mismatch"] = True
                # is the source index itself a substitute? (stale index check on the common configuration)
                stale_equal = 0
                for img, sv in zip(images[:5], src_vectors["common"][:5]):
                    b = to_device(t_common_proc.process_images([img]))
                    stale_equal += int(bitwise(forward_vectors(tmodel, b), sv))
            row["native"] = native_pages
            row["common"] = common_pages
            row["forced_reuse_control"] = forced
            row["source_vectors_equal_target_pages_of_5"] = stale_equal
            if not model_level_ok:
                row["verdict"] = "REJECT_VISUAL_PATH"
                row["reason"] = "visual-branch parameters differ as loaded: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
            elif common_pages["bitwise"] == len(images):
                if native_pages["bitwise"] == len(images):
                    row["verdict"] = "ACCEPT"
                    row["reason"] = "model-level and page-level checks pass natively; reconstruction bitwise on all pages"
                else:
                    row["verdict"] = "ACCEPT_COMMON_CONFIG"
                    row["reason"] = f"native processor differs: {native_pages['rejected']}/{len(images)} pages rejected at the boundary; exact under the common configuration"
            else:
                row["verdict"] = "MISMATCH_AFTER_CHECKS"
                row["reason"] = f"checks passed but {len(images) - common_pages['bitwise']} pages differ (max err {common_pages['max_err']})"
        except Exception as e:  # noqa: BLE001
            row.update({"verdict": "RUN_ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}"})
            (args.out / f"run-error-{rel['repo'].replace('/', '__')}.txt").write_text(traceback.format_exc())
        row["seconds"] = time.perf_counter() - t0
        with rows_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row[k] for k in row if k not in ("processor",)}, default=str), flush=True)
        del tmodel, tbase
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
