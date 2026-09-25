"""Read-only processor audit before expensive encoding; no quality claims."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from colpali_engine.models import ColQwen2_5_Processor
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    native = ColQwen2_5_Processor.from_pretrained(
        cfg["native_processor"], local_files_only=True
    )
    common = ColQwen2_5_Processor.from_pretrained(
        cfg["processor_model"], local_files_only=True
    )
    common.image_processor.max_pixels = 602112
    common.image_processor.size["longest_edge"] = 602112
    pages = pq.read_table(cfg["hr_corpus"]).to_pylist()[:8]
    records = []
    for page in pages:
        image = Image.open(io.BytesIO(page["image"]["bytes"])).convert("RGB")
        a, b = native.process_images([image]), common.process_images([image])
        records.append(
            {
                "id": page["corpus_id"],
                "image_sha256": hashlib.sha256(page["image"]["bytes"]).hexdigest(),
                "native_grid": a["image_grid_thw"].tolist(),
                "common_grid": b["image_grid_thw"].tolist(),
                "native_tokens": a["image_grid_thw"].prod().item() // 4,
                "common_tokens": b["image_grid_thw"].prod().item() // 4,
                "pixel_equal": torch.equal(a["pixel_values"], b["pixel_values"]),
            }
        )
    queries = pq.read_table(
        Path(cfg["hr_corpus"]).parents[1] / "queries/test-00000-of-00001.parquet"
    ).to_pylist()[:8]
    query_same = []
    for q in queries:
        a, b = (
            native.process_queries([q["query"]]),
            common.process_queries([q["query"]]),
        )
        query_same.append(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a))
    result = {
        "native_processor": native.to_dict(),
        "native_image_processor": native.image_processor.to_dict(),
        "common_processor": common.to_dict(),
        "common_image_processor": common.image_processor.to_dict(),
        "pages": records,
        "query_inputs_equal": query_same,
    }
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {
                "page_tokens": [
                    (r["native_tokens"], r["common_tokens"]) for r in records
                ],
                "query_inputs_equal": query_same,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
