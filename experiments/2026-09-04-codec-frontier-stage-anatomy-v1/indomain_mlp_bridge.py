#!/usr/bin/env python3
"""In-domain Drift-Adapter residual MLP bridge on the frozen holdouts.

The paper's residual-MLP bridge was fitted on ViDoRe Energy query tokens and
evaluated cross-domain. A reviewer can argue that handicaps the baseline. Here
the same recipe (hidden 256, dropout 0.1, AdamW 3e-4 / wd 0.01, token batches of
256, up to 50 epochs, patience 5 on a 20% validation split) is fitted on the
non-holdout 70% queries of each corpus and evaluated on that corpus's holdout
against the legacy (v0.1) document bank. Reference ranking: exact v0.2 index.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

CORPORA = ("arxivqa", "docvqa", "flickr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-code-root", type=Path, required=True)
    p.add_argument("--support-code-root", type=Path, required=True)
    p.add_argument("--arxivqa-parquet", type=Path, required=True)
    p.add_argument("--docvqa-parquet", type=Path, required=True)
    p.add_argument("--flickr-root", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--processor-model", type=Path, required=True)
    p.add_argument("--old-adapter", type=Path, required=True)
    p.add_argument("--new-adapter", type=Path, required=True)
    p.add_argument("--projection", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    return p.parse_args()


RECIPE = {"hidden_units": 256, "dropout": 0.1, "learning_rate": 3e-4, "weight_decay": 0.01,
          "batch_size_tokens": 256, "maximum_epochs": 50, "early_stopping_patience": 5}


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one GPU")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path.insert(0, str(args.matrix_code_root)); sys.path.insert(0, str(args.support_code_root))
    from run_matrix import FlickrRows, ParquetRows, metric_rows, paired_bootstrap
    from run_smoke import load_compatible_adapter, load_projection
    from streaming_maxsim import streaming_maxsim

    torch.manual_seed(args.seed)
    projection = load_projection(args.projection, torch)

    def load_model(adapter: Path) -> Any:
        base = ColQwen2_5.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
        with torch.no_grad():
            base.custom_text_proj.weight.copy_(projection["custom_text_proj.weight"].to(base.custom_text_proj.weight))
            base.custom_text_proj.bias.copy_(projection["custom_text_proj.bias"].to(base.custom_text_proj.bias))
        return load_compatible_adapter(base, adapter).to(args.device).eval()

    old_model, new_model = load_model(args.old_adapter), load_model(args.new_adapter)
    processor = ColQwen2_5_Processor.from_pretrained(args.processor_model, local_files_only=True)
    datasets = {"arxivqa": ParquetRows(args.arxivqa_parquet, "query"), "docvqa": ParquetRows(args.docvqa_parquet, "query"), "flickr": FlickrRows(args.flickr_root)}

    def encode_queries(texts: list[str]) -> tuple[list[Any], list[Any]]:
        olds, news = [], []
        with torch.inference_mode():
            for s in range(0, len(texts), 32):
                qb = processor.process_queries(texts[s : s + 32]).to(args.device)
                ov, nv = old_model(**qb).float(), new_model(**qb).float()
                for o, n, m in zip(ov, nv, qb["attention_mask"], strict=True):
                    k = m.bool(); olds.append(o[k].cpu()); news.append(n[k].cpu())
        return olds, news

    def encode_docs(model: Any, ds: Any, ids: list[int]) -> list[Any]:
        out = []
        with torch.inference_mode():
            for i in ids:
                b = processor.process_images([ds.get(i)[0]]).to(args.device)
                v = model(**b).float()
                out.append(v[0][b["attention_mask"][0].bool()].to(torch.bfloat16).cpu())
        return out

    class ResidualMLP(torch.nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.first = torch.nn.Linear(dimension, RECIPE["hidden_units"])
            self.dropout = torch.nn.Dropout(RECIPE["dropout"])
            self.second = torch.nn.Linear(RECIPE["hidden_units"], dimension)
            self.diagonal = torch.nn.Parameter(torch.ones(dimension))

        def forward(self, x: Any) -> Any:
            return F.normalize(self.diagonal * (x + self.second(self.dropout(F.gelu(self.first(x))))), dim=-1)

    results = {}
    for corpus in args.corpora:
        ds = datasets[corpus]
        rows = len(ds)
        perm = np.random.RandomState(0).permutation(rows)
        holdout = sorted(int(v) for v in perm[: int(round(0.3 * rows))])
        fit_ids = [i for i in range(rows) if i not in set(holdout)]
        if args.smoke:
            holdout, fit_ids = holdout[: args.smoke], fit_ids[: max(16, args.smoke)]
        rng = np.random.RandomState(args.seed)
        val_mask = rng.rand(len(fit_ids)) < 0.2
        fit_old, fit_new = encode_queries([ds.get(i)[1] for i in fit_ids])
        x_tr = torch.cat([n for n, v in zip(fit_new, val_mask) if not v]); y_tr = torch.cat([o for o, v in zip(fit_old, val_mask) if not v])
        x_va = torch.cat([n for n, v in zip(fit_new, val_mask) if v]); y_va = torch.cat([o for o, v in zip(fit_old, val_mask) if v])

        mlp = ResidualMLP(x_tr.shape[1]).to(args.device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
        best, best_state, stale, history = float("inf"), None, 0, []
        t0 = time.perf_counter()
        for epoch in range(RECIPE["maximum_epochs"]):
            mlp.train()
            perm_t = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(args.seed + epoch))
            losses = []
            for s in range(0, len(perm_t), RECIPE["batch_size_tokens"]):
                sel = perm_t[s : s + RECIPE["batch_size_tokens"]]
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(mlp(x_tr[sel].to(args.device)), y_tr[sel].to(args.device))
                loss.backward(); opt.step(); losses.append(float(loss))
            mlp.eval()
            with torch.inference_mode():
                va = float(F.mse_loss(mlp(x_va.to(args.device)), y_va.to(args.device)))
            history.append({"epoch": epoch + 1, "train_mse": statistics.fmean(losses), "validation_mse": va})
            if va < best - 1e-9:
                best, best_state, stale = va, {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= RECIPE["early_stopping_patience"]:
                    break
        mlp.load_state_dict(best_state); mlp.eval()
        fit_seconds = time.perf_counter() - t0
        with torch.inference_mode():
            cos_before = float(F.cosine_similarity(x_va, y_va, dim=-1).mean())
            cos_after = float(F.cosine_similarity(mlp(x_va.to(args.device)).cpu(), y_va, dim=-1).mean())

        # holdout evaluation
        _, hold_new = encode_queries([ds.get(i)[1] for i in holdout])
        with torch.inference_mode():
            hold_mapped = [mlp(q.to(args.device)).cpu() for q in hold_new]
        old_docs = encode_docs(old_model, ds, holdout)
        new_docs = encode_docs(new_model, ds, holdout)
        scores = {"target_full": streaming_maxsim(hold_new, new_docs, device=args.device, query_chunk=32, document_chunk=16),
                  "stale": streaming_maxsim(hold_new, old_docs, device=args.device, query_chunk=32, document_chunk=16),
                  "mlp_indomain": streaming_maxsim(hold_mapped, old_docs, device=args.device, query_chunk=32, document_chunk=16)}
        target_order = np.argsort(-scores["target_full"], axis=1)
        agg, rows_by = {}, {}
        for m, s in scores.items():
            agg[m], rows_by[m] = metric_rows(s, target_order)
        boot = {m: {f: paired_bootstrap(rows_by[m], rows_by["target_full"], field=f, draws=20_000, seed=args.seed + 7 * i + j)
                    for j, f in enumerate(("ndcg_at_5", "top10_overlap"))} for i, m in enumerate(("stale", "mlp_indomain"))}
        results[corpus] = {"fit_queries": len(fit_ids), "train_tokens": int(len(x_tr)), "validation_tokens": int(len(x_va)),
                           "epochs": len(history), "best_validation_mse": best, "fit_seconds": fit_seconds,
                           "validation_token_cosine_before": cos_before, "validation_token_cosine_after": cos_after,
                           "metrics": agg, "differences_vs_target_full": {m: {"ndcg_at_5": agg[m]["ndcg_at_5"] - agg["target_full"]["ndcg_at_5"]} for m in ("stale", "mlp_indomain")},
                           "paired_bootstrap_vs_target_full": boot, "history": history}
        print(json.dumps({corpus: {m: {"ta10": agg[m]["top10_overlap"], "ndcg5": agg[m]["ndcg_at_5"]} for m in agg}, "cos": [cos_before, cos_after]}), flush=True)
        (args.output_root / f"{corpus}.json").write_text(json.dumps(results[corpus], indent=2))
    (args.output_root / "result.json").write_text(json.dumps({"recipe": RECIPE, "seed": args.seed, "results": results}, indent=2))


if __name__ == "__main__":
    main()
