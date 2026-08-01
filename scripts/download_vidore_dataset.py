#!/usr/bin/env python3
"""Download only the public parquet components of a ViDoRe v3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="e.g. vidore/vidore_v3_finance_en",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", help="optional immutable dataset revision")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[
            "corpus/*.parquet",
            "queries/*.parquet",
            "qrels/*.parquet",
        ],
        local_dir=args.output,
    )
    print(root)


if __name__ == "__main__":
    main()
