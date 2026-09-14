"""Reconstruct target representations; reject invalid states and encode raw pages."""

import argparse
import json
from pathlib import Path

from _common import contract, digest, setup, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Target key in the config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int)
    args = parser.parse_args()
    from PIL import Image

    from reprforge.integrations.colqwen import replay

    config = json.loads(args.config.read_text())
    source = json.loads((args.source_index / "manifest.json").read_text())
    torch, processor, model, base = setup(
        config,
        config["targets"][args.target],
        args.max_pixels if args.max_pixels is not None else source["max_pixels"],
    )
    identity = contract(base, processor)
    valid_contract = identity == source["contract"]
    args.output.mkdir(parents=True, exist_ok=False)
    bank, routes = [], []
    with torch.inference_mode():
        for page in source["pages"]:
            path = Path(page["path"])
            state_path = args.source_index / "states" / page["state"]
            valid = (
                valid_contract
                and digest(path) == page["sha256"]
                and state_path.is_file()
                and digest(state_path) == page["state_sha256"]
            )
            if valid:
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                vectors = replay(base, state, torch)
            else:
                with Image.open(path) as image:
                    batch = processor.process_images([image.convert("RGB")]).to(
                        "cuda:0"
                    )
                vectors = model(**batch)[0][batch["attention_mask"][0].bool()]
            bank.append(vectors.cpu())
            routes.append("replay" if valid else "raw")
    torch.save(bank, args.output / "representations.pt")
    write(
        args.output / "manifest.json",
        {
            "target": args.target,
            "contract": identity,
            "pages": source["pages"],
            "routes": routes,
            "source_contract_valid": valid_contract,
        },
    )


if __name__ == "__main__":
    main()
