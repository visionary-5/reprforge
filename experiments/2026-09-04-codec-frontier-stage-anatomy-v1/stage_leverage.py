#!/usr/bin/env python3
"""Backbone-agnostic cut-leverage measurement.

For a ColPali-family retriever, time per page (CUDA-synchronised):
  preprocess  -- processor.process_images on CPU + host-to-device copy
  vision      -- forward time inside the vision tower module (forward hooks)
  total       -- complete document forward (vision + merger/projector + LM + head)
Leverage = (preprocess + vision) / (preprocess + total). It upper-bounds the
saving any post-vision replay can achieve on that backbone, so it predicts
whether a semantic cut is worth materialising before any codec is fitted.

Families: colpali (PaliGemma), colsmol (Idefics3), colqwen2.5 (Qwen2.5-VL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("colpali", "colsmol", "colqwen2.5"), required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--processor-model", type=Path, default=None)
    p.add_argument("--matrix-code-root", type=Path, required=True)
    p.add_argument("--arxivqa-parquet", type=Path, required=True)
    p.add_argument("--docvqa-parquet", type=Path, required=True)
    p.add_argument("--flickr-root", type=Path, required=True)
    p.add_argument("--pages", type=int, default=40)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    import numpy as np
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path.insert(0, str(args.matrix_code_root))
    from run_matrix import FlickrRows, ParquetRows

    from colpali_engine.models import ColIdefics3, ColIdefics3Processor, ColPali, ColPaliProcessor, ColQwen2_5, ColQwen2_5_Processor
    cls, proc_cls = {"colpali": (ColPali, ColPaliProcessor), "colsmol": (ColIdefics3, ColIdefics3Processor), "colqwen2.5": (ColQwen2_5, ColQwen2_5_Processor)}[args.family]
    model = cls.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    model.load_adapter(str(args.adapter))
    model = model.to(args.device).eval()
    processor = proc_cls.from_pretrained(args.processor_model or args.adapter, local_files_only=True)

    # locate the vision tower module
    vision_name = None
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("vision_tower", "vision_model", "visual") and "merger" not in name:
            vision_name = name
            break
    if vision_name is None:
        raise RuntimeError("could not locate vision module")
    vision_module = dict(model.named_modules())[vision_name]
    vision_params = sum(p.numel() for p in vision_module.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    marks: dict[str, float] = {}

    def pre_hook(module: Any, inputs: Any) -> None:
        torch.cuda.synchronize(); marks["vision_start"] = time.perf_counter()

    def post_hook(module: Any, inputs: Any, output: Any) -> None:
        torch.cuda.synchronize(); marks["vision_end"] = time.perf_counter()

    vision_module.register_forward_pre_hook(pre_hook)
    vision_module.register_forward_hook(post_hook)

    datasets = {"arxivqa": ParquetRows(args.arxivqa_parquet, "query"), "docvqa": ParquetRows(args.docvqa_parquet, "query"), "flickr": FlickrRows(args.flickr_root)}
    result: dict[str, Any] = {"family": args.family, "base_model": str(args.base_model), "adapter": str(args.adapter),
                              "vision_module": vision_name, "vision_params": vision_params, "total_params": total_params,
                              "hardware": torch.cuda.get_device_name(0), "pages_per_corpus": args.pages, "corpora": {}}
    for corpus, ds in datasets.items():
        rows = len(ds)
        permutation = np.random.RandomState(0).permutation(rows)
        holdout = sorted(int(v) for v in permutation[: int(round(0.3 * rows))])[: args.pages + args.warmup]
        per_page = []
        with torch.inference_mode():
            for k, idx in enumerate(holdout):
                image = ds.get(idx)[0]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                batch = processor.process_images([image]).to(args.device)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                out = model(**batch)
                torch.cuda.synchronize(); t2 = time.perf_counter()
                if k < args.warmup:
                    continue
                vision = marks["vision_end"] - marks["vision_start"]
                per_page.append({"preprocess": t1 - t0, "vision": vision, "total_forward": t2 - t1,
                                 "doc_vectors": int(out.shape[1]), "seq_tokens": int(batch["attention_mask"].sum())})
        agg = {k: float(np.mean([p[k] for p in per_page])) for k in ("preprocess", "vision", "total_forward", "doc_vectors", "seq_tokens")}
        agg["leverage"] = (agg["preprocess"] + agg["vision"]) / (agg["preprocess"] + agg["total_forward"])
        agg["vision_share_of_forward"] = agg["vision"] / agg["total_forward"]
        agg["pages"] = len(per_page)
        result["corpora"][corpus] = agg
        print(json.dumps({corpus: agg}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
