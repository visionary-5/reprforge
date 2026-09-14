"""Reconstruct the historical 580-page distractor input from public MMDocIR pages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from support.files import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent /
        "quality/mmdocir-distractors.json").read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.pages_parquet) != manifest["source_sha256"]:
        raise ValueError("Not the pinned public MMDocIR_pages.parquet")
    import pyarrow.parquet as pq
    table = pq.read_table(args.pages_parquet, columns=["image_binary"])
    selected = table.take([r["source_row"] for r in manifest["rows"]])
    for row, blob in zip(manifest["rows"], selected.column(0).to_pylist()):
        if hashlib.sha256(blob).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed at source row {row['source_row']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, args.output)
    print("580 image payloads verified; Parquet container stores only image_binary.")


if __name__ == "__main__":
    main()
