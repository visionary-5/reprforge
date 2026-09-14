"""Prepare public model/data inputs and verify their content hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from files import sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    sources = json.loads(Path(__file__).with_name('sources.json').read_text())
    destination = args.root.resolve()
    if not args.check_only:
        from huggingface_hub import hf_hub_download
        if destination.exists() and not args.resume:
            raise FileExistsError('Use a new root, --resume, or --check-only')
        destination.mkdir(parents=True, exist_ok=True)
        for item in sources:
            target = destination / item['destination']
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if 'bundled' in item:
                    shutil.copyfile(ROOT / item['bundled'], target)
                else:
                    paths = [hf_hub_download(item['repo'], name, revision=item['revision'],
                                            repo_type=item.get('repo_type', 'model'))
                             for name in item.get('filenames', [item.get('filename')])]
                    if len(paths) == 1:
                        shutil.copyfile(paths[0], target)
                    else:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        if pa.__version__ != '19.0.1':
                            raise RuntimeError('TAT-DQA preparation requires pyarrow==19.0.1')
                        pq.write_table(pa.concat_tables([pq.read_table(p) for p in paths]), target)
            if sha256(target) != item['sha256']:
                raise ValueError(f'Input checksum mismatch: {target}')
    for item in sources:
        if sha256(destination / item['destination']) != item['sha256']:
            raise ValueError(f"Input checksum mismatch: {item['destination']}")
    config = json.loads((ROOT / 'configs/colqwen.json').read_text())
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, str) and value.startswith('inputs/'):
            return str(destination / value.removeprefix('inputs/'))
        return value
    config_path = destination / 'endpoint-config.json'
    if not config_path.exists() and not args.check_only:
        config_path.write_text(json.dumps(resolve(config), indent=2) + '\n')
    print(f'All inputs verified. Config: {config_path}')


if __name__ == '__main__':
    main()
