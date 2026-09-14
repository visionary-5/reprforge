"""Encode page images and persist their post-merger states and representations."""

import argparse
import json
from pathlib import Path

from _common import contract, digest, setup, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pages",
        type=Path,
        required=True,
        help="JSON list of image paths, relative to this JSON file",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=12845056)
    args = parser.parse_args()
    from PIL import Image

    from reprforge.integrations.colqwen import capture, replay

    config = json.loads(args.config.read_text())
    pages = [
        (args.pages.parent / p).resolve() for p in json.loads(args.pages.read_text())
    ]
    if not pages or any(not p.is_file() for p in pages):
        raise ValueError("Provide a nonempty list of existing image files")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "states").mkdir()
    torch, processor, _, base = setup(config, config["source"], args.max_pixels)
    identity = contract(base, processor)
    records, bank = [], []
    with torch.inference_mode():
        for i, path in enumerate(pages):
            with Image.open(path) as image:
                batch = processor.process_images([image.convert("RGB")]).to("cuda:0")
            state = capture(base, batch, torch)
            bank.append(replay(base, state, torch).cpu())
            state_path = args.output / "states" / f"{i:06d}.pt"
            torch.save(state, state_path)
            records.append(
                {
                    "path": str(path),
                    "sha256": digest(path),
                    "state": state_path.name,
                    "state_sha256": digest(state_path),
                }
            )
    torch.save(bank, args.output / "representations.pt")
    write(
        args.output / "manifest.json",
        {
            "contract": identity,
            "pages": records,
            "max_pixels": args.max_pixels,
            "representation_format": "ordered page token matrices; no ANN backend",
        },
    )


if __name__ == "__main__":
    main()
