"""Download pinned public weights/data and restore frozen execution metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_artifacts import ROOT, extract, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="New external input directory")
    parser.add_argument("--check-only", action="store_true", help="Verify existing inputs without network access")
    parser.add_argument("--resume", action="store_true", help="Resume missing downloads, retaining verified existing files")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sources = json.loads((here / "sources.json").read_text())
    metadata = json.loads((ROOT / "docs/input-metadata-manifest.json").read_text())
    destination = args.root.resolve()
    if not args.check_only:
        from huggingface_hub import hf_hub_download
        if destination.exists() and not args.resume:
            raise FileExistsError("Use a new root, --resume, or --check-only")
        bundle = next(x for x in json.loads((ROOT / "evidence/bundles.json").read_text())
                      if x["kind"] == "input-metadata")
        archive = ROOT / "evidence" / bundle["file"]
        assert sha256(archive) == bundle["sha256"]
        if destination.exists():
            for name, digest in metadata.items():
                assert sha256(destination / name) == digest, name
        else:
            extract(archive, destination)
        for item in sources:
            target = destination / item["destination"]
            if target.exists():
                assert sha256(target) == item["sha256"], item["destination"]
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            paths = [hf_hub_download(item["repo"], name, revision=item["revision"],
                                    repo_type=item.get("repo_type", "model"))
                     for name in item.get("filenames", [item.get("filename")])]
            if len(paths) == 1:
                shutil.copyfile(paths[0], target)
            else:
                import pyarrow as pa
                import pyarrow.parquet as pq
                if pa.__version__ != "19.0.1":
                    raise RuntimeError("TAT-DQA byte-identical preparation requires pyarrow==19.0.1")
                pq.write_table(pa.concat_tables([pq.read_table(p) for p in paths]), target)
            assert sha256(target) == item["sha256"], item["destination"]
    for item in sources:
        assert sha256(destination / item["destination"]) == item["sha256"], item["destination"]
    for name, digest in metadata.items():
        assert sha256(destination / name) == digest, name
    config = json.loads((here / "config.example.json").read_text())
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, str) and value.startswith("inputs/"):
            return str(destination / value.removeprefix("inputs/"))
        return value
    config_path = destination / "endpoint-config.json"
    if not config_path.exists() and not args.check_only:
        config_path.write_text(json.dumps(resolve(config), indent=2) + "\n")
    print(f"All inputs verified. Config: {config_path}")


if __name__ == "__main__":
    main()
