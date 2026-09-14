"""Certify identical logical Energy inputs and issue an explicit reproduction protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "support"))
from files import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    import pyarrow.parquet as pq
    here = Path(__file__).resolve().parent
    payload_path = here / "energy-endpoint-payload.json"
    expected = json.loads(payload_path.read_text())
    observed = [{"corpus_id": str(r["corpus_id"]),
                 "image_sha256": hashlib.sha256(r["image_binary"]).hexdigest()}
                for r in pq.read_table(args.slice, columns=["corpus_id", "image_binary"]).to_pylist()]
    if observed != expected["rows"]:
        raise ValueError("Reconstructed page count, order, IDs or image bytes differ")
    original = here.parent / "reconstruction/official_upgrade/gpu-protocol.json"
    protocol = json.loads(original.read_text())
    assert protocol["frozen_inputs"]["dataset_sha256"] == expected["original_file_sha256"]
    receipt = {
        "original_protocol_sha256": sha256(original),
        "original_dataset_sha256": expected["original_file_sha256"],
        "reconstructed_dataset_sha256": sha256(args.slice),
        "payload_manifest_sha256": sha256(payload_path),
        "ordered_equal_pages": len(observed),
        "scope": "All input image bytes and corpus IDs equal in order; Parquet container bytes may differ.",
        "changes": "Only dataset file digest amended for explicit reproduction; no model, page, batch, metric or timing definition changed.",
        "gpu_execution": "Not performed by this utility; original measurement files unchanged.",
    }
    protocol["protocol_id"] += "-certified-reconstruction"
    protocol["frozen_inputs"]["dataset_sha256"] = receipt["reconstructed_dataset_sha256"]
    protocol["reproduction_input_certificate"] = receipt
    args.output_root.mkdir(parents=True)
    for filename, value in (("input-equivalence.json", receipt), ("gpu-protocol.json", protocol)):
        (args.output_root / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(f"Certified {len(observed)} ordered pages. Original protocol and results unchanged.")


if __name__ == "__main__":
    main()
