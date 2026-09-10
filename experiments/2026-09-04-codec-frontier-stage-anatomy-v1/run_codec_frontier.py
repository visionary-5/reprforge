#!/usr/bin/env python3
"""Codec frontier and build-cost anatomy for the ColQwen2.5 v0.1->v0.2 transition.

Two questions the SIGIR draft leaves open:

1. Fidelity--storage frontier of the semantic cut. How much of the gap between
   ReprForge (PCA-256/INT8) and exact target reconstruction is codec loss? We
   replay the target suffix from an exact BF16 cut (TA should be ~1.0) and from
   INT8, INT4, and nested PCA ranks {512,256,128,64} with INT8 coefficients, all
   on the same frozen 30% document-disjoint holdouts used by the paper.

2. Where does raw build time go? Per page we time, with CUDA synchronisation,
   (a) image preprocessing on CPU, (b) the visual prefix (embedding lookup,
   vision tower, merger, scatter, RoPE index), and (c) the language-model
   suffix plus terminal projection. The ratio (a+b)/(a+b+c) is the "cut
   leverage" that upper-bounds any post-vision replay saving.

The runner reuses dataset readers, metric code and the paired bootstrap from
the frozen sigir-version-evolution-matrix-v1 code so that numbers are directly
comparable with Table 2 of the draft.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

CORPORA = ("arxivqa", "docvqa", "flickr", "infovqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matrix-code-root", type=Path, required=True)
    parser.add_argument("--support-code-root", type=Path, required=True)
    parser.add_argument("--arxivqa-parquet", type=Path, required=True)
    parser.add_argument("--docvqa-parquet", type=Path, required=True)
    parser.add_argument("--flickr-root", type=Path, required=True)
    parser.add_argument("--infovqa-parquet", type=Path, default=None)
    parser.add_argument("--family", choices=("colqwen2.5", "colqwen2"), default="colqwen2.5")
    parser.add_argument("--projection-format", choices=("header-bin", "safetensors"), default="header-bin")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-model", type=Path, required=True)
    parser.add_argument("--old-adapter", type=Path, required=True)
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    return parser.parse_args()


# --------------------------------------------------------------------------- codecs

def codec_rank(codec: str) -> int | None:
    if codec.startswith("pca"):
        return int(codec.removeprefix("pca").split("_", 1)[0])
    return None


def encode_decode(values: Any, codec: str, torch: Any, mean: Any, basis: Any) -> tuple[Any, int]:
    """Round-trip one page's visual states [tokens, channels] through a codec.

    Returns the restored BF16 tensor and the exact number of payload bytes that
    a compact serialisation would need (excluding the shared basis and mean,
    which are collection-scoped and reported separately).
    """
    tokens, channels = values.shape
    x = values.float()
    if codec == "bf16":
        return x.to(torch.bfloat16), tokens * channels * 2
    if codec == "int8_token":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(x / scale).clamp(-127, 127)
        return (q * scale).to(torch.bfloat16), tokens * channels + tokens * 2
    if codec == "int4_group64":
        groups = channels // 64
        g = x.reshape(tokens, groups, 64)
        scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        q = torch.round(g / scale).clamp(-7, 7)
        return (q * scale).reshape(tokens, channels).to(torch.bfloat16), tokens * channels // 2 + tokens * groups * 2
    rank = codec_rank(codec)
    if rank is None:
        raise ValueError(codec)
    b = basis[:, :rank]
    coefficients = (x - mean) @ b
    if codec.endswith("_bf16"):
        restored = coefficients.to(torch.bfloat16).float() @ b.T + mean
        return restored.to(torch.bfloat16), tokens * rank * 2
    scale = coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(coefficients / scale).clamp(-127, 127)
    restored = (q * scale) @ b.T + mean
    return restored.to(torch.bfloat16), tokens * rank + tokens * 2


# --------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    if args.family == "colqwen2.5":
        from colpali_engine.models import ColQwen2_5 as ModelClass
        from colpali_engine.models import ColQwen2_5_Processor as ProcessorClass
    else:
        from colpali_engine.models import ColQwen2 as ModelClass
        from colpali_engine.models import ColQwen2Processor as ProcessorClass

    sys.path.insert(0, str(args.matrix_code_root))
    sys.path.insert(0, str(args.support_code_root))
    from run_matrix import FlickrRows, ParquetRows, metric_rows, paired_bootstrap, sha256
    from run_smoke import load_projection as load_projection_header
    from streaming_maxsim import streaming_maxsim

    def load_projection(path: Path) -> dict[str, Any]:
        if args.projection_format == "header-bin":
            return load_projection_header(path, torch)
        from safetensors.torch import load_file

        tensors = load_file(str(path))
        return {"custom_text_proj.weight": tensors["custom_text_proj.weight"],
                "custom_text_proj.bias": tensors["custom_text_proj.bias"]}

    def load_compatible_adapter(base: Any, adapter: Path) -> Any:
        """Load LoRA keys saved under the pre-4.53 module tree onto the renamed tree.

        Qwen2-VL and Qwen2.5-VL moved decoder layers from ``model.layers`` to
        ``language_model.layers``; without the remap PEFT silently drops every
        decoder LoRA tensor. The count of remapped tensors is recorded, not
        hard-coded, so both families load through one path.
        """

        from peft import PeftConfig, get_peft_model
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        config = PeftConfig.from_pretrained(adapter, local_files_only=True)
        model = get_peft_model(base, config)
        weights = load_peft_weights(adapter, device="cpu", local_files_only=True)
        old_prefix, new_prefix = "base_model.model.model.layers.", "base_model.model.language_model.layers."
        remapped = {(new_prefix + k[len(old_prefix):] if k.startswith(old_prefix) else k): v for k, v in weights.items()}
        result = set_peft_model_state_dict(model, remapped, adapter_name="default")
        missing = [k for k in result.missing_keys if "lora_" in k]
        unexpected = [k for k in result.unexpected_keys if "lora_" in k]
        if missing or unexpected:
            raise RuntimeError(f"adapter load incomplete: missing={len(missing)} unexpected={len(unexpected)}")
        model.adapter_load_diagnostics = {"checkpoint_tensors": len(weights),
                                          "remapped_decoder_lora_tensors": sum(k.startswith(old_prefix) for k in weights)}
        return model.eval()

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen-before-gpu-output":
        raise RuntimeError("protocol is not frozen")
    codecs: list[str] = protocol["codecs"]
    max_rank = max([r for r in (codec_rank(c) for c in codecs) if r is not None] + [1])
    expected = protocol["transition"]
    for path, value in (
        (args.old_adapter / "adapter_model.safetensors", expected["old_adapter_sha256"]),
        (args.new_adapter / "adapter_model.safetensors", expected["new_adapter_sha256"]),
        (args.projection, expected["projection_seed_sha256"]),
    ):
        if sha256(path) != value:
            raise RuntimeError(f"artifact hash mismatch: {path}")

    torch.manual_seed(int(protocol["torch_seed"]))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch 2.5.0's cuDNN SDPA backend can emit NaN for masked vision attention; the paper ran on 2.5.1.
    torch.backends.cuda.enable_cudnn_sdp(False)
    projection = load_projection(args.projection)

    def load_model(adapter: Path) -> tuple[Any, Any]:
        base = ModelClass.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
        )
        with torch.no_grad():
            base.custom_text_proj.weight.copy_(projection["custom_text_proj.weight"].to(base.custom_text_proj.weight))
            base.custom_text_proj.bias.copy_(projection["custom_text_proj.bias"].to(base.custom_text_proj.bias))
        wrapped = load_compatible_adapter(base, adapter).to(args.device).eval()
        return wrapped, wrapped.get_base_model()

    old_model, old_base = load_model(args.old_adapter)
    new_model, new_base = load_model(args.new_adapter)
    processor = ProcessorClass.from_pretrained(args.processor_model, local_files_only=True)

    def sync() -> None:
        torch.cuda.synchronize()

    def prefix(base: Any, batch: Any) -> dict[str, Any]:
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        grid = batch["image_grid_thw"]
        offsets = grid[:, 1] * grid[:, 2]
        pixels = torch.cat([row[: int(o.item())] for row, o in zip(batch["pixel_values"], offsets, strict=True)], dim=0)
        embeddings = base.get_input_embeddings()(input_ids)
        vision = torch.cat(base.get_image_features(pixels, grid), dim=0).to(embeddings.device, embeddings.dtype)
        if not torch.isfinite(vision).all():
            raise RuntimeError(f"non-finite vision features: pixels finite={bool(torch.isfinite(pixels).all())}, grid={grid.tolist()}")
        embeddings = embeddings.masked_scatter(
            (input_ids == int(base.config.image_token_id)).unsqueeze(-1).expand_as(embeddings), vision
        )
        position_ids, _ = base.get_rope_index(input_ids, grid, None, attention_mask=attention)
        return {
            "attention": attention,
            "embeds": embeddings,
            "position_ids": position_ids,
            "cache_position": torch.arange(embeddings.shape[1], device=args.device),
            "valid": [torch.where(row.bool())[0] for row in attention],
            "visual": [
                torch.where((ids == int(base.config.image_token_id)) & mask.bool())[0]
                for ids, mask in zip(input_ids, attention, strict=True)
            ],
        }

    def suffix(base: Any, data: dict[str, Any], embeds: Any) -> list[Any]:
        output = base.language_model(
            input_ids=None, inputs_embeds=embeds, attention_mask=data["attention"],
            position_ids=data["position_ids"], cache_position=data["cache_position"],
            use_cache=False, output_hidden_states=False, return_dict=True,
        )
        projected = functional.normalize(base.custom_text_proj(output.last_hidden_state), dim=-1).float()
        return [projected[row].index_select(0, valid) for row, valid in enumerate(data["valid"])]

    all_datasets = {
        "arxivqa": ParquetRows(args.arxivqa_parquet, "query"),
        "docvqa": ParquetRows(args.docvqa_parquet, "query"),
        "flickr": FlickrRows(args.flickr_root),
    }
    if args.infovqa_parquet is not None:
        all_datasets["infovqa"] = ParquetRows(args.infovqa_parquet, "query")
    results: dict[str, Any] = {"family": args.family, "adapter_load": {
        "old": old_model.adapter_load_diagnostics, "new": new_model.adapter_load_diagnostics}}
    for corpus in args.corpora:
        dataset = all_datasets[corpus]
        rows = len(dataset)
        permutation = np.random.RandomState(0).permutation(rows)
        evaluation = sorted(int(v) for v in permutation[: int(round(0.3 * rows))])
        calibration = [i for i in range(rows) if i not in set(evaluation)][:64]
        if args.smoke:
            evaluation = evaluation[: args.smoke]
            calibration = calibration[: min(args.smoke, len(calibration))]
        corpus_root = args.output_root / corpus
        corpus_root.mkdir()

        # ---- calibration: one nested PCA basis of rank max_rank, query-free
        cal_started = time.perf_counter()
        observed = []
        with torch.inference_mode():
            for start in range(0, len(calibration), args.image_batch_size):
                images = [dataset.get(i)[0] for i in calibration[start : start + args.image_batch_size]]
                data = prefix(old_base, processor.process_images(images).to(args.device))
                for row, visual in enumerate(data["visual"]):
                    observed.append(data["embeds"][row].index_select(0, visual).detach().cpu())
        observed_t = torch.cat(observed, dim=0)
        token_count = min(int(protocol["sampled_visual_tokens"]), len(observed_t))
        selection = torch.linspace(0, len(observed_t) - 1, token_count).round().long()
        cal = observed_t.index_select(0, selection).to(args.device).float()
        mean = cal.mean(dim=0)
        rank = min(max_rank, len(cal) - 1)
        # Exact PCA through the (channels x channels) covariance in float64: deterministic, well conditioned,
        # and the eigenvectors sorted by descending eigenvalue give a nested basis for every rank.
        centered = (cal - mean).double()
        if not torch.isfinite(centered).all():
            raise RuntimeError("non-finite visual states in calibration sample")
        covariance = (centered.T @ centered).cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        basis = eigenvectors[:, :rank].float().to(args.device)
        sync()
        calibration_seconds = time.perf_counter() - cal_started
        explained = eigenvalues.clamp_min(0).cumsum(0) / eigenvalues.clamp_min(0).sum()
        torch.save({"mean": mean.to(torch.bfloat16).cpu(), "basis": basis.to(torch.bfloat16).cpu()}, corpus_root / "basis.pt")

        # ---- canary: old and new prefixes must be bitwise equal
        first = processor.process_images([dataset.get(evaluation[0])[0]]).to(args.device)
        with torch.inference_mode():
            a, b = prefix(old_base, first), prefix(new_base, first)
        if not torch.equal(a["embeds"], b["embeds"]):
            raise RuntimeError("prefix canary failed")

        # ---- encode holdout documents
        banks: dict[str, list[Any]] = {"target_full": [], "stale": [], **{c: [] for c in codecs}}
        stage = {"preprocess": 0.0, "prefix": 0.0, "suffix_target": 0.0, "suffix_old": 0.0}
        replay_seconds = {c: 0.0 for c in codecs}
        payload_bytes = {c: 0 for c in codecs}
        visual_tokens = 0
        sequence_tokens = 0
        terminal_tokens = 0
        per_page: list[dict[str, Any]] = []
        with torch.inference_mode():
            for start in range(0, len(evaluation), args.image_batch_size):
                indices = evaluation[start : start + args.image_batch_size]
                images = [dataset.get(i)[0] for i in indices]

                sync(); t0 = time.perf_counter()
                batch = processor.process_images(images).to(args.device)
                sync(); t1 = time.perf_counter()
                data = prefix(new_base, batch)
                sync(); t2 = time.perf_counter()
                target = suffix(new_base, data, data["embeds"])
                sync(); t3 = time.perf_counter()
                old = suffix(old_base, data, data["embeds"])
                sync(); t4 = time.perf_counter()
                stage["preprocess"] += t1 - t0
                stage["prefix"] += t2 - t1
                stage["suffix_target"] += t3 - t2
                stage["suffix_old"] += t4 - t3
                banks["target_full"].extend(v.to(torch.bfloat16).cpu() for v in target)
                banks["stale"].extend(v.to(torch.bfloat16).cpu() for v in old)

                for codec in codecs:
                    sync(); s0 = time.perf_counter()
                    embeds = data["embeds"].clone()
                    for row, visual in enumerate(data["visual"]):
                        restored, nbytes = encode_decode(data["embeds"][row].index_select(0, visual), codec, torch, mean, basis)
                        embeds[row, visual] = restored
                        payload_bytes[codec] += nbytes
                    replayed = suffix(new_base, data, embeds)
                    sync()
                    replay_seconds[codec] += time.perf_counter() - s0
                    banks[codec].extend(v.to(torch.bfloat16).cpu() for v in replayed)

                for row, visual in enumerate(data["visual"]):
                    nvis, nseq, nterm = len(visual), int(data["attention"][row].sum()), len(data["valid"][row])
                    visual_tokens += nvis; sequence_tokens += nseq; terminal_tokens += nterm
                    per_page.append({"row_id": indices[row], "visual_tokens": nvis, "sequence_tokens": nseq,
                                     "preprocess_s": (t1 - t0) / len(indices), "prefix_s": (t2 - t1) / len(indices),
                                     "suffix_s": (t3 - t2) / len(indices)})
                print(json.dumps({"corpus": corpus, "encoded": min(start + len(indices), len(evaluation)), "total": len(evaluation)}), flush=True)

        # ---- queries: holdout queries under both encoders, plus in-domain bridge fitting
        def encode_queries(texts: list[str]) -> tuple[list[Any], list[Any]]:
            olds, news = [], []
            with torch.inference_mode():
                for start in range(0, len(texts), args.query_batch_size):
                    qb = processor.process_queries(texts[start : start + args.query_batch_size]).to(args.device)
                    ov, nv = old_model(**qb).float(), new_model(**qb).float()
                    for o, n, mask in zip(ov, nv, qb["attention_mask"], strict=True):
                        keep = mask.bool()
                        olds.append(o[keep].detach().cpu()); news.append(n[keep].detach().cpu())
            return olds, news

        queries = [dataset.get(i)[1] for i in evaluation]
        _, new_queries = encode_queries(queries)

        # In-domain compatibility bridges (Drift-Adapter style, new query space -> old query space),
        # fitted on token pairs from the non-holdout 70% queries of the same corpus. No holdout text is used.
        holdout_set = set(evaluation)
        fit_ids = [i for i in range(rows) if i not in holdout_set]
        if args.smoke:
            fit_ids = fit_ids[: max(8, args.smoke)]
        fit_old, fit_new = encode_queries([dataset.get(i)[1] for i in fit_ids])
        x = torch.cat(fit_new, dim=0).to(args.device).double()   # source: target-version query tokens
        y = torch.cat(fit_old, dim=0).to(args.device).double()   # target: legacy query space
        u, _, vt = torch.linalg.svd(x.T @ y)
        procrustes = (u @ vt).float().cpu()
        xa = torch.cat((x, torch.ones((x.shape[0], 1), device=args.device, dtype=x.dtype)), dim=1)
        affine = torch.linalg.solve(xa.T @ xa + 0.01 * torch.eye(xa.shape[1], device=args.device, dtype=x.dtype), xa.T @ y).float().cpu()

        def bridge(values: list[Any], kind: str) -> list[Any]:
            out = []
            for v in values:
                src = v.float()
                mapped = src @ procrustes if kind == "procrustes" else torch.cat((src, torch.ones((src.shape[0], 1))), dim=1) @ affine
                out.append(functional.normalize(mapped, dim=-1))
            return out

        query_banks = {"procrustes_indomain": bridge(new_queries, "procrustes"), "affine_indomain": bridge(new_queries, "affine")}
        banks["procrustes_indomain"] = banks["stale"]
        banks["affine_indomain"] = banks["stale"]

        methods = ["target_full", "stale", "procrustes_indomain", "affine_indomain", *codecs]
        scores = {m: streaming_maxsim(query_banks.get(m, new_queries), banks[m], device=args.device, query_chunk=32, document_chunk=16) for m in methods}
        target_order = np.argsort(-scores["target_full"], axis=1)
        aggregates, rows_by_method = {}, {}
        for m in methods:
            aggregates[m], rows_by_method[m] = metric_rows(scores[m], target_order)
        # exact top-1 agreement and full-ranking Kendall-style statistic are cheap extras
        for m in methods:
            order = np.argsort(-scores[m], axis=1)
            aggregates[m]["top1_agreement"] = float(np.mean(order[:, 0] == target_order[:, 0]))
        bootstrap = {
            m: {f: paired_bootstrap(rows_by_method[m], rows_by_method["target_full"], field=f, draws=20_000,
                                    seed=int(protocol["torch_seed"]) + 100 * CORPORA.index(corpus) + 10 * methods.index(m) + fi)
                for fi, f in enumerate(("ndcg_at_5", "top10_overlap"))}
            for m in methods if m != "target_full"
        }
        pages = len(evaluation)
        channels = int(mean.shape[0])
        terminal_dim = int(banks["target_full"][0].shape[-1])
        result = {
            "corpus": corpus,
            "hardware": torch.cuda.get_device_name(torch.cuda.current_device()),
            "scope": {"rows": rows, "holdout_pages": pages, "calibration_pages": len(calibration), "image_batch_size": args.image_batch_size},
            "tokens": {"visual_per_page": visual_tokens / pages, "sequence_per_page": sequence_tokens / pages,
                       "terminal_per_page": terminal_tokens / pages, "hidden_channels": channels, "terminal_dim": terminal_dim},
            "stage_seconds_total": stage,
            "stage_seconds_per_page": {k: v / pages for k, v in stage.items()},
            "cut_leverage": (stage["preprocess"] + stage["prefix"]) / (stage["preprocess"] + stage["prefix"] + stage["suffix_target"]),
            "calibration_seconds": calibration_seconds,
            "pca_explained_variance": {str(r): float(explained[r - 1]) for r in (32, 64, 128, 256, 512) if r <= rank},
            "replay_seconds_total": replay_seconds,
            "replay_seconds_per_page": {c: v / pages for c, v in replay_seconds.items()},
            "payload_bytes_per_page": {c: v / pages for c, v in payload_bytes.items()},
            "terminal_bytes_per_page": {"float32": terminal_tokens / pages * terminal_dim * 4, "bf16": terminal_tokens / pages * terminal_dim * 2},
            "basis_bytes": (corpus_root / "basis.pt").stat().st_size,
            "indomain_bridge_fit": {"queries": len(fit_ids), "token_pairs": int(x.shape[0]), "ridge_lambda": 0.01},
            "metrics": aggregates,
            "differences_vs_target_full": {m: {"ndcg_at_5": aggregates[m]["ndcg_at_5"] - aggregates["target_full"]["ndcg_at_5"],
                                              "top10_overlap": aggregates[m]["top10_overlap"] - aggregates["target_full"]["top10_overlap"]}
                                          for m in methods if m != "target_full"},
            "paired_bootstrap_vs_target_full": bootstrap,
            "per_page": per_page,
        }
        (corpus_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results[corpus] = {k: v for k, v in result.items() if k != "per_page"}
        print(json.dumps({"completed": corpus, "cut_leverage": result["cut_leverage"],
                          "ta10": {m: aggregates[m]["top10_overlap"] for m in methods}}, indent=2), flush=True)

    summary = {"protocol_id": protocol["protocol_id"], "protocol_sha256": sha256(args.protocol), "smoke": args.smoke,
               "family": args.family, "adapter_load": results.pop("adapter_load"), "results": {k: v for k, v in results.items() if k != "family"}}
    (args.output_root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
