"""Verify published source hashes and LFS evidence, optionally extracting safely."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as target:
                import shutil
                shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    records = json.loads((ROOT / "docs/provenance.json").read_text())
    for row in records:
        if "published" in row:
            assert sha256(ROOT / row["published"]) == row["sha256"], row["published"]
    bundles = json.loads((ROOT / "evidence/bundles.json").read_text())
    for bundle in bundles:
        path = ROOT / "evidence" / bundle["file"]
        if path.stat().st_size < 200 and path.read_bytes().startswith(b"version https://git-lfs"):
            raise RuntimeError("LFS pointer found: run git lfs pull first")
        assert sha256(path) == bundle["sha256"], str(path)
        if args.extract_to and bundle["kind"] in {"historical-evidence", "independent-endpoint-evidence"}:
            extract(path, args.extract_to)
    if args.extract_to:
        for row in records:
            if "artifact" in row:
                assert sha256(args.extract_to / row["artifact"]) == row["sha256"], row["artifact"]
        endpoint = args.extract_to / "independent-endpoint"
        result = json.loads((endpoint / "result.json").read_text())
        assert result == json.loads((ROOT / "experiments/independent-endpoint/summary.json").read_text())
        manifest = json.loads((endpoint / "manifest.json").read_text())
        for source, digest in manifest["code_sha256"].items():
            local = (ROOT / "experiments/independent-endpoint/run.py" if source.endswith("/independent-endpoint/run.py")
                     else ROOT / "experiments/support" / Path(source).name)
            assert sha256(local) == digest, str(local)
        for target, summary in result.items():
            pages = [json.loads(line) for line in (endpoint / f"{target}-pages.jsonl").read_text().splitlines()]
            ranks = json.loads((endpoint / f"{target}-rankings.json").read_text())
            assert len(pages) == summary["pages"]
            assert sum(p["tensor_equal"] for p in pages) == summary["tensor_equal_pages"]
            assert sum(p["elements"] for p in pages) == summary["elements"]
            assert sum(p["equal_elements"] for p in pages) == summary["equal_elements"]
            assert len(ranks["raw_top10"]) == len(ranks["replay_top10"]) == summary["queries"]
            assert sum(a == b for a, b in zip(ranks["raw_top10"], ranks["replay_top10"])) == summary["ordered_top10_equal_queries"]
    print("Published code and evidence hashes verified.")


if __name__ == "__main__":
    main()
