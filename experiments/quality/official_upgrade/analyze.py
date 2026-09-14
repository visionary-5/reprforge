"""Rebuild official-upgrade metrics with the shared valid-query evaluator."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Extracted raw output directory')
    parser.add_argument('--query-mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # The shared evaluator expects the mask next to its input directory.
    # Copy the raw files into a new output workspace, never modify evidence.
    local = args.output / 'input'
    local.mkdir()
    for name in ['result.json', 'per_query.json', 'rankings.json']:
        shutil.copyfile(args.input / name, local / name)
    shutil.copyfile(args.query_mask, args.output / 'query_mask.json')
    path = Path(__file__).parents[2] / 'quality/analyze.py'
    spec = importlib.util.spec_from_file_location('pooled_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.load(local)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'table.md').write_text(module.matrix_md(result) + '\n')
    (args.output / 'by_collection.md').write_text(module.by_collection_md(result) + '\n')


if __name__ == '__main__':
    main()
