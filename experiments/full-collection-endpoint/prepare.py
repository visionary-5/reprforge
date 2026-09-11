"""Freeze complete collection inputs before GPU execution; no new data selection."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    manifest = json.loads((args.prior / 'manifest.json').read_text())
    config = manifest['config']
    config['targets'] = {'vidore-v0.2': config['targets']['vidore-v0.2']}
    seen = set()
    counts = {}
    for name, spec in config['collections'].items():
        before = len(seen)
        for row in pq.read_table(spec['parquet'], columns=['image']).to_pylist():
            value = row['image']
            blob = value['bytes'] if isinstance(value, dict) else value
            seen.add(hashlib.sha256(blob).hexdigest())
        counts[name] = len(seen) - before
    protocol = json.loads((args.prior / 'protocol.json').read_text())
    protocol.update(id='full-collection-independent-v1', pages=len(seen),
        selection='Every unique image-byte SHA256 in the six existing collections; deterministic round-robin order.',
        queries='Original fixed query holdouts; all collection pages in gallery, no added distractors.',
        claim='Independent full-collection representation and ordered ranking fidelity; paired document encoding time.',
        compare_stale_index=True,
        scope='Six complete collections, one named target; no new dataset or family. No physical index equality claim.')
    protocol['targets'] = {'vidore-v0.2': protocol['targets']['vidore-v0.2']}
    for name, data in [('config.json', config), ('protocol.json', protocol), ('counts.json', counts)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'pages':len(seen), 'by_collection':counts}))


if __name__ == '__main__':
    main()
