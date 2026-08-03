#!/usr/bin/env python3
"""Freeze the rendered-page union needed by an IRPAPERS candidate surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.run_pairwise_admission_physical import _candidate_surface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    candidates, *_ = _candidate_surface(
        corpus_ids,
        np.asarray(surface["bm25_scores"], dtype=np.float64),
        np.asarray(surface["visual_scores"], dtype=np.float64),
        candidate_k=args.candidate_k,
        cutoff=min(5, args.candidate_k - 1),
    )
    selected_ids = sorted(
        {corpus_ids[int(position)] for position in candidates.flat},
        key=lambda value: tuple(int(part) for part in value.split("_", maxsplit=1)),
    )
    pages = []
    for item_id in selected_ids:
        pdf_id, page_number = item_id.split("_", maxsplit=1)
        pages.append(
            {
                "dataset_id": item_id,
                "pdf_id": int(pdf_id),
                "page_number": int(page_number),
            }
        )
    payload = {
        "schema_version": 1,
        "source": str(args.score_surface.resolve()),
        "candidate_k": args.candidate_k,
        "counts": {"target_pages": len(pages)},
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
