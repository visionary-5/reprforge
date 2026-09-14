"""Compare persisted tensor payload bits on CPU, including the sign of zero."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    import torch
    torch.set_num_threads(4)
    results = json.loads((args.output_root / "result.json").read_text())
    report = {}
    for target in results:
        path = args.output_root / f"{target}-banks.pt"
        bank = torch.load(path, map_location="cpu", weights_only=True)
        assert len(bank["raw"]) == len(bank["replay"]) == results[target]["pages"]
        pages = 0
        payload_bytes = 0
        for raw, replay in zip(bank["raw"], bank["replay"]):
            assert raw.dtype == replay.dtype == torch.bfloat16
            assert raw.shape == replay.shape
            a, b = raw.contiguous().view(torch.uint8), replay.contiguous().view(torch.uint8)
            pages += int(torch.equal(a, b))
            payload_bytes += a.numel()
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
        report[target] = {"pages": len(bank["raw"]), "bitwise_equal_pages": pages,
                          "tensor_payload_bytes_per_route": payload_bytes,
                          "bank_file_sha256": digest.hexdigest(),
                          "dtype": "bfloat16",
                          "scope": "Persisted raw/replay tensor payloads; not physical index or torch.save file equality"}
        del bank
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
