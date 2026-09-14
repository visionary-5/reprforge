"""Compare generated page representations against an independent raw reference."""

import argparse
from pathlib import Path

from _common import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="torch.save list of native raw page token matrices",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="torch.save list with the same page IDs and order",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Expected equal, nonzero page counts in identical order")
    rows = []
    for a, b in zip(reference, candidate, strict=False):
        same_shape = a.shape == b.shape
        rows.append(
            {
                "shape_equal": same_shape,
                "dtype_equal": a.dtype == b.dtype,
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                "elementwise_equal": bool(a.dtype == b.dtype and torch.equal(a, b)),
                "max_abs_error": float((a.float() - b.float()).abs().max())
                if same_shape
                else None,
            }
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {"pages": len(rows), "comparisons": rows})


if __name__ == "__main__":
    main()
