"""Independent pytrec_eval check of saved rankings under the frozen tie rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytrec_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    d = args.directory
    queries = json.loads((d / "queries.json").read_text())
    pages = json.loads((d / "pages.json").read_text())
    qrels = json.loads((d / "qrels.json").read_text())
    checks = {}
    for route in ["native", "common"]:
        scores = np.load(d / f"{route}-scores.npy")
        ranks = np.argsort(-scores, axis=1, kind="stable")
        # Rank-only monotone scores ensure pytrec_eval uses our explicit tie rule.
        predictions = {
            q["id"]: {
                str(pages[j]["id"]): float(len(pages) - rank)
                for rank, j in enumerate(ranks[i])
            }
            for i, q in enumerate(queries)
        }
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.5,10"})
        observed = evaluator.evaluate(predictions)
        expected = json.loads((d / f"{route}-per-query.json").read_text())
        errors = [
            abs(observed[r["id"]][f"ndcg_cut_{k}"] - r[f"ndcg_at_{k}"])
            for r in expected
            for k in [5, 10]
        ]
        assert max(errors) < 1e-12, max(errors)
        checks[route] = {
            "queries": len(observed),
            "max_metric_error": max(errors),
            "mean_ndcg_at_5": float(
                np.mean([v["ndcg_cut_5"] for v in observed.values()])
            ),
            "mean_ndcg_at_10": float(
                np.mean([v["ndcg_cut_10"] for v in observed.values()])
            ),
        }
    (d / "independent-metrics.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
