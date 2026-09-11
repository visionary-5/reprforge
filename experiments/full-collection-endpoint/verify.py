"""Verify completed expanded evidence without assuming positive outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def equal(a,b,torch):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def verify_core(root,tensors):
    result=read(root/'result.json'); pages=lines(root/'pages.jsonl')
    n=result['pages_per_case']; cases=result['protocol']['cases']
    assert len(pages)==result['rows']==n*len(cases)
    for case in cases:
        subset=[r for r in pages if r['case']==case]
        assert [r['index'] for r in subset]==list(range(n))
        assert len({r['page']['image_sha256'] for r in subset})==n
        for k,v in result['by_case'][case].items():
            assert sum(r['checks'][k]['bit_equal'] for r in subset)==v
    identities={case:[(r['page'],r['state_sha256']) for r in pages if r['case']==case] for case in cases}
    assert all(v==next(iter(identities.values())) for v in identities.values())
    assert result['all_native_called_vision']==all(r['raw_vision_calls']>0 for r in pages)
    assert result['all_replay_skipped_vision']==all(r['replay_vision_calls']==0 for r in pages)
    assert result['all_checks_finite']==all(c['finite'] for r in pages for c in r['checks'].values())
    verified=0
    if tensors:
        import torch
        assert digest(root/'banks.pt')==result['banks_sha256']
        bank=torch.load(root/'banks.pt',weights_only=True,map_location='cpu')
        assert len(bank)==len(pages)
        for record,payload in zip(pages,bank,strict=True):
            assert (record['case'],record['index'])==(payload['case'],payload['index'])
            for name,check in record['checks'].items():
                value=payload['replay'] if name=='factorized_replay' else payload['extra']
                assert equal(payload['raw'],value,torch)==check['bit_equal']
                assert bool(torch.isfinite(payload['raw']).all() and torch.isfinite(value).all())==check['finite']
                if check['shape_equal']:
                    assert float((payload['raw'].float()-value.float()).abs().max())==check['max_abs_error']
                verified+=1
    return dict(kind='mechanism',pages_per_condition=n,conditions=len(cases),rows=len(pages),
        tensor_comparisons_verified=verified,outcomes=result['by_case'])


def verify_full(root,tensors):
    result=read(root/'vidore-v0.2-result.json'); pages=lines(root/'vidore-v0.2-pages.jsonl')
    inputs=read(root/'inputs.json'); source=read(root/'source.json');ranks=read(root/'vidore-v0.2-rankings.json')
    assert len(pages)==len(inputs['pages'])==len(source['states'])==result['pages']
    assert len({p['image_sha256'] for p in pages})==len(pages)
    for page,item in zip(pages,inputs['pages'],strict=True):
        assert all(page[k]==v for k,v in item.items())
    assert result['source_pid']==source['pid'] and result['pid']!=source['pid']
    for k,r in [('tensor_equal','tensor_equal_pages'),('finite','finite_pages'),('elements','elements'),('equal_elements','equal_elements')]:
        assert sum(p[k] for p in pages)==result[r]
    for key in ['raw_page_seconds','replay_with_read_seconds']:
        assert abs(sum(p[key] for p in pages)-result[key])<1e-8
    assert len(ranks['raw_top10'])==len(ranks['replay_top10'])==len(inputs['queries'])==result['queries']
    for bank in ranks.values():
        assert all(len(row)==10 and len(set(row))==10 and all(0<=x<len(pages) for x in row) for row in bank)
    ordered=sum(a==b for a,b in zip(ranks['raw_top10'],ranks['replay_top10'],strict=True))
    assert ordered==result['ordered_top10_equal_queries']
    verified=0
    if tensors:
        import torch
        bank=torch.load(root/'vidore-v0.2-banks.pt',weights_only=True,map_location='cpu')
        assert len(bank['raw'])==len(bank['replay'])==len(pages)
        for page,a,b in zip(pages,bank['raw'],bank['replay'],strict=True):
            assert equal(a,b,torch)==page['bit_equal']
            assert a.numel()==page['elements']
            verified+=1
    return dict(kind='full-collection',pages=len(pages),queries=len(inputs['queries']),
        byte_equal_pages=sum(p['bit_equal'] for p in pages),ordered_top10_equal=ordered,
        tensor_comparisons_verified=verified)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--kind',choices=['mechanism','full'],required=True)
    ap.add_argument('--tensors',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=(verify_core if args.kind=='mechanism' else verify_full)(args.run,args.tensors)
    report['input_sha256']={p.name:digest(p) for p in args.run.iterdir() if p.suffix in ['.json','.jsonl'] and p.resolve()!=args.output.resolve()}
    report['verifier_sha256']=digest(Path(__file__))
    if args.tensors:
        report['tensor_bank_sha256']={p.name:digest(p) for p in args.run.glob('*banks.pt')}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
