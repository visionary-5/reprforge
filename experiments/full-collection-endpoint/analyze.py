"""Analyze completed native endpoints, preserving all pages and frozen queries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    root=args.run
    result=json.loads((root/'vidore-v0.2-result.json').read_text())
    ranks=json.loads((root/'stale-rankings.json').read_text())
    inputs=json.loads((root/'inputs.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    import pyarrow.parquet as pq
    by_hash={p['image_sha256']:i for i,p in enumerate(inputs['pages'])}
    gold={}
    for name,spec in manifest['config']['collections'].items():
        for i,row in enumerate(pq.read_table(spec['parquet'],columns=['image']).to_pylist()):
            val=row['image'];blob=val['bytes'] if isinstance(val,dict) else val
            gold[(name,i)]=by_hash[hashlib.sha256(blob).hexdigest()]
    assert result['pages']==len(by_hash)
    records=[]
    for route,key in [('stale','stale_top10'),('raw','target_top10'),('replay','replay_top10')]:
        assert len(ranks[key])==len(inputs['queries'])
        for i,(q,ranking,reference) in enumerate(zip(inputs['queries'],ranks[key],ranks['target_top10'],strict=True)):
            g=gold[(q['collection'],q['row'])]
            position=ranking.index(g)+1 if g in ranking else 11
            records.append(dict(route=route,query=i,collection=q['collection'],gold=g,
                ndcg5=float(1/np.log2(position+1)) if position<=5 else 0.,recall10=int(position<=10),
                ta10=len(set(ranking)&set(reference))/10,ordered1=int(ranking[0]==reference[0]),
                ordered10=int(ranking==reference)))
    metrics=['ndcg5','recall10','ta10','ordered1','ordered10']
    summary={}
    for route in ['stale','raw','replay']:
        subset=[r for r in records if r['route']==route]
        summary[route]={k:float(np.mean([r[k] for r in subset])) for k in metrics}
    (root/'analysis.json').write_text(json.dumps(dict(pages=result['pages'],queries=len(inputs['queries']),
        summary=summary,state_storage_bytes=sum(s['bytes'] for s in json.loads((root/'source.json').read_text())['states']),scope='Frozen ViDoRe evaluation splits, not full original source benchmarks; descriptive fixed-set evaluation.'),indent=2)+'\n')
    (root/'per-query-analysis.json').write_text(json.dumps(records)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
