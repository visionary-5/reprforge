#!/usr/bin/env python3
"""Record durable physical evidence before a reproducible index is released."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wall_seconds(path: Path) -> float:
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = float(fields[1])
    if "real" not in values:
        raise ValueError(f"GNU time file lacks real seconds: {path}")
    return values["real"]


def index_inventory(index: Path) -> tuple[int, int]:
    if not index.is_dir() or index.is_symlink():
        raise ValueError(f"index must be a real directory: {index}")
    files = 0
    total_bytes = 0
    for root, directories, names in os.walk(index, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *names]:
            item = root_path / name
            if item.is_symlink():
                raise ValueError(f"index contains a symbolic link: {item}")
        for name in names:
            item = root_path / name
            if item.is_file():
                files += 1
                total_bytes += item.stat().st_size
    if files == 0 or total_bytes == 0:
        raise ValueError(f"index is empty: {index}")
    return files, total_bytes


def ranking_inventory(path: Path) -> tuple[int, int]:
    query_ids: set[str] = set()
    rows = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"invalid ranking row {path}:{line_number}")
            query_ids.add(fields[0])
            float(fields[2])
            rows += 1
    if not rows:
        raise ValueError(f"ranking is empty: {path}")
    return len(query_ids), rows


def ranking_path(case_root: Path) -> Path:
    candidates = (
        case_root / "full/result/ranking.txt",
        case_root / "full/result-ranking-top100/ranking.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[0])


def build_receipt(case_root: Path, selection_manifest: Path | None) -> dict[str, Any]:
    case_root = case_root.resolve(strict=True)
    paths = {
        "run_manifest": case_root / "run-manifest.json",
        "build_time": case_root / "timing/full-build.time",
        "eval_time": case_root / "timing/full-eval.time",
        "ranking": ranking_path(case_root),
        "build_log": case_root / "full/build.log",
        "eval_log": case_root / "full/eval.log",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    index = case_root / "full/index"
    index_files, index_bytes = index_inventory(index)
    ranking_queries, ranking_rows = ranking_inventory(paths["ranking"])
    artifacts = {
        name: {
            "relative_path": str(path.relative_to(case_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }
    selection = None
    if selection_manifest is not None:
        selection_manifest = selection_manifest.resolve(strict=True)
        selection = {
            "path": str(selection_manifest),
            "bytes": selection_manifest.stat().st_size,
            "sha256": sha256(selection_manifest),
        }
    return {
        "schema_version": 1,
        "status": "complete_and_safe_to_release_reproducible_index",
        "case_name": case_root.name,
        "case_root": str(case_root),
        "timing_seconds": {
            "direct_build_wall": wall_seconds(paths["build_time"]),
            "evaluation_wall": wall_seconds(paths["eval_time"]),
        },
        "physical_index": {"bytes": index_bytes, "files": index_files},
        "ranking": {"queries": ranking_queries, "rows": ranking_rows},
        "artifacts": artifacts,
        "selection_manifest": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.case_root / "case-receipt.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = build_receipt(args.case_root, args.selection_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
