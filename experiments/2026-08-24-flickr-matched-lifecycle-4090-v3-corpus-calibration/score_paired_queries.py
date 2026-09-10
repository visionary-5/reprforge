#!/usr/bin/env python3
"""Recover paired Flickr query outcomes from the physical matched indexes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = {
    "raw_full": ("raw-cache/flickr/ps.pt", "raw_full"),
    "raw_light10": ("indexes/raw-light-10.pt", "raw_light10"),
    "raw_light05": ("indexes/raw-light-05.pt", "raw_light05"),
    "reprforge_full": ("replay-cache/flickr/ps.pt", "reprforge_full"),
    "reprforge_light10": ("indexes/replay-light-10.pt", "reprforge_light10"),
    "reprforge_light05": ("indexes/replay-light-05.pt", "reprforge_light05"),
}
CONTRASTS = {
    "full": ("reprforge_full", "raw_full"),
    "light10": ("reprforge_light10", "raw_light10"),
    "light05": ("reprforge_light05", "raw_light05"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import numpy as np
    import torch
    from scipy.stats import rankdata, wilcoxon

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "post-hoc-after-aggregate":
        raise RuntimeError("paired analysis protocol is not frozen")
    root = args.experiment_root
    meta = json.loads((root / "raw-cache/flickr/meta.json").read_text())
    replay_meta = json.loads((root / "replay-cache/flickr/meta.json").read_text())
    if meta != replay_meta or meta["slice"] != "flickr" or meta["n"] != 1000:
        raise RuntimeError("unexpected or mismatched cache metadata")
    queries = torch.load(root / "raw-cache/flickr/qs.pt", weights_only=True)

    by_document: dict[int, list[int]] = {}
    for query_index, document in enumerate(meta["gold"]):
        by_document.setdefault(int(document), []).append(query_index)
    documents = sorted(by_document)
    permutation = np.random.RandomState(0).permutation(len(documents))
    test_documents = [documents[int(index)] for index in permutation[:300]]
    column = {document: index for index, document in enumerate(test_documents)}
    evaluation_queries = [
        query_index
        for document in test_documents
        for query_index in by_document[document]
    ]
    if len(evaluation_queries) != 300:
        raise RuntimeError(f"expected 300 evaluation queries, got {len(evaluation_queries)}")
    gold_columns = torch.tensor(
        [column[int(meta["gold"][index])] for index in evaluation_queries]
    )
    selected = [queries[index] for index in evaluation_queries]
    maximum = max(int(query.shape[0]) for query in selected)
    channels = int(selected[0].shape[1])
    query_pad = torch.zeros(
        len(selected), maximum, channels, dtype=torch.bfloat16, device=args.device
    )
    query_mask = torch.zeros(len(selected), maximum, device=args.device)
    for row, query in enumerate(selected):
        query_pad[row, : query.shape[0]] = query.to(args.device).to(torch.bfloat16)
        query_mask[row, : query.shape[0]] = 1.0

    outcomes: dict[str, dict[str, list[float] | list[int] | float]] = {}
    for method, (relative_path, _) in METHODS.items():
        pages = torch.load(root / relative_path, weights_only=True)
        if len(pages) != 1000:
            raise RuntimeError(f"{method} has {len(pages)} pages")
        scores = torch.empty(len(selected), len(test_documents), device=args.device)
        with torch.inference_mode():
            for document in test_documents:
                page = pages[document].to(args.device).to(torch.bfloat16)
                token_scores = (query_pad @ page.T).max(dim=2)[0]
                scores[:, column[document]] = (token_scores.float() * query_mask).sum(dim=1)
        order = torch.argsort(scores.cpu(), dim=1, descending=True)
        ranks, values = [], []
        for row in range(len(evaluation_queries)):
            rank = int((order[row] == gold_columns[row]).nonzero(as_tuple=True)[0].item()) + 1
            ranks.append(rank)
            values.append(1.0 / math.log2(rank + 1) if rank <= 5 else 0.0)
        outcomes[method] = {"ranks": ranks, "ndcg_at_5": values, "mean": float(np.mean(values))}
        del pages, scores, order
        torch.cuda.empty_cache()
        print(json.dumps({"method": method, "ndcg_at_5": outcomes[method]["mean"]}), flush=True)

    stats = json.loads((root / "analysis-output/stats.json").read_text())
    expected_means = {row["method"]: float(row["ndcg_at_5"]) for row in stats["rows"]}
    fidelity = {
        method: float(outcomes[method]["mean"]) - expected_means[method]
        for method in METHODS
    }
    if any(abs(value) > 0.0001 for value in fidelity.values()):
        raise RuntimeError(f"paired reconstruction disagrees with aggregate scorer: {fidelity}")

    rng = np.random.default_rng(int(protocol["bootstrap"]["seed"]))
    draws = int(protocol["bootstrap"]["draws"])
    bootstrap_indices = rng.integers(0, len(evaluation_queries), size=(draws, len(evaluation_queries)))
    contrasts = {}
    raw_pvalues = {}
    for name, (left, right) in CONTRASTS.items():
        difference = np.asarray(outcomes[left]["ndcg_at_5"]) - np.asarray(outcomes[right]["ndcg_at_5"])
        boot = difference[bootstrap_indices].mean(axis=1)
        nonzero = difference[difference != 0]
        if len(nonzero):
            test = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
            ranks = rankdata(np.abs(nonzero))
            positive = float(ranks[nonzero > 0].sum())
            negative = float(ranks[nonzero < 0].sum())
            rank_biserial = (positive - negative) / (positive + negative)
            pvalue = float(test.pvalue)
        else:
            positive = negative = rank_biserial = 0.0
            pvalue = 1.0
        raw_pvalues[name] = pvalue
        contrasts[name] = {
            "mean_difference": float(difference.mean()),
            "bootstrap_95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
            "wins_ties_losses": [int((difference > 0).sum()), int((difference == 0).sum()), int((difference < 0).sum())],
            "nonzero_pairs": int(len(nonzero)),
            "wilcoxon_w_plus": positive,
            "wilcoxon_w_minus": negative,
            "wilcoxon_p_two_sided": pvalue,
            "rank_biserial": rank_biserial,
        }
    adjusted = holm(raw_pvalues)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value

    result = {
        "protocol": protocol["protocol_id"],
        "scope": {"test_documents": len(test_documents), "evaluation_queries": len(evaluation_queries)},
        "methods": {method: {"ndcg_at_5": value["mean"]} for method, value in outcomes.items()},
        "contrasts": contrasts,
        "aggregate_fidelity": fidelity,
        "gates": {
            "aggregate_fidelity": all(abs(value) <= 0.0001 for value in fidelity.values()),
            "light10_paired_noninferiority": contrasts["light10"]["bootstrap_95"][0] >= -0.01,
        },
        "boundary": protocol["boundary"],
    }
    args.output.mkdir(parents=True)
    (args.output / "paired-result.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["query_index", "gold_document"] + [f"{method}_rank" for method in METHODS] + [f"{method}_ndcg_at_5" for method in METHODS]
    with (args.output / "paired-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, query_index in enumerate(evaluation_queries):
            record = {"query_index": query_index, "gold_document": int(meta["gold"][query_index])}
            for method in METHODS:
                record[f"{method}_rank"] = outcomes[method]["ranks"][row]
                record[f"{method}_ndcg_at_5"] = outcomes[method]["ndcg_at_5"][row]
            writer.writerow(record)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
