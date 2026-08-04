#!/usr/bin/env python3
"""Attach paired query confidence intervals to residual-witness curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reprforge.heterogeneity_atlas import paired_bootstrap_ci, query_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = np.load(args.full_runtime, allow_pickle=False)
    pooled = np.load(args.pooled_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    relevance = [dict() for _ in full["query_ids"]]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    baseline = {
        "full": query_metrics(full["scores"], tuple(relevance), ks=(5, 10, 100)),
        "pool9": query_metrics(
            pooled["scores"], tuple(relevance), ks=(5, 10, 100)
        ),
    }
    reports = {}
    for path in args.result:
        payload = json.loads(path.read_text())
        for name, row in payload["reports"].items():
            key = f"{path.stem}:{name}"
            metrics = {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in row["per_query_witness"].items()
            }
            reports[key] = {
                "source": str(path),
                "configuration": name,
                "token_fraction": row["mean_token_fraction"],
                "metrics": {
                    metric: {
                        "mean": float(values.mean()),
                        "vs_full": paired_bootstrap_ci(
                            values, baseline["full"][metric]
                        ),
                        "vs_pool9": paired_bootstrap_ci(
                            values, baseline["pool9"][metric]
                        ),
                    }
                    for metric, values in metrics.items()
                },
            }
    result = {
        "protocol": "paired query bootstrap; 4000 resamples",
        "baselines": {
            name: {
                metric: float(values.mean())
                for metric, values in metrics.items()
                if metric in {"ndcg_at_5", "ndcg_at_10", "recall_at_100"}
            }
            for name, metrics in baseline.items()
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
