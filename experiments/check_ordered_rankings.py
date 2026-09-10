"""Check ordered top-k equality in saved pooled rankings; no GPU or new scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check(path):
    data = json.loads(path.read_text())
    keep = [i for i, name in enumerate(data['collection']) if name != 'shiftproject']
    rows = {}
    for target, routes in data['top20'].items():
        if 'bf16' not in routes:
            rows[target] = {'status': 'BF16 route not measured'}
            continue
        ref, route = routes['target_full'], routes['bf16']
        rows[target] = {
            'queries': len(keep),
            'ordered_top1_equal': sum(ref[i][:1] == route[i][:1] for i in keep),
            'ordered_top10_equal': sum(ref[i][:10] == route[i][:10] for i in keep),
            'ordered_top20_equal': sum(ref[i][:20] == route[i][:20] for i in keep),
            'scope': 'Saved shared-prefix target_full versus BF16 suffix route; not independent raw encoding',
        }
    return {'rankings_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'gallery_pages': len(data['gallery']), 'targets': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rankings', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump({str(path): check(path) for path in a.rankings}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
