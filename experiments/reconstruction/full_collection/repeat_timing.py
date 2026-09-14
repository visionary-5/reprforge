"""Fresh-process paired timing after full-collection correctness verification."""
import argparse
import importlib.util
from importlib.metadata import version
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--verification',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--round',type=int)
    args=ap.parse_args()
    protocol=json.loads(Path(__file__).with_name('repeat-protocol.json').read_text())
    verification=json.loads(args.verification.read_text())
    for name in ['manifest.json','inputs.json']:
        if verification['input_sha256'][name]!=hashlib.sha256((args.prior/name).read_bytes()).hexdigest():
            raise ValueError('Verification report belongs to different inputs')
    if args.round is not None and not 0<=args.round<protocol['rounds']:
        raise ValueError('Round is outside the frozen protocol')
    if verification['pages']!=verification['tensor_comparisons_verified'] or verification['byte_equal_pages']!=verification['pages']:
        raise ValueError('Resolve incomplete verification or fidelity failures before timing repetitions')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit GPU selection required')
    if args.round is None:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for i in range(protocol['rounds']):
            subprocess.run([sys.executable,__file__,'--prior',str(args.prior),'--verification',str(args.verification),
                            '--output',str(args.output/f'round-{i}'),'--round',str(i)],check=True)
        summaries=[json.loads((args.output/f'round-{i}'/'result.json').read_text()) for i in range(protocol['rounds'])]
        (args.output/'result.json').write_text(json.dumps(summaries,indent=2)+'\n')
        return
    args.output.mkdir(parents=True,exist_ok=False)
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from PIL import Image
    spec=importlib.util.spec_from_file_location('endpoint',Path(__file__).with_name('run.py'))
    ep=importlib.util.module_from_spec(spec);spec.loader.exec_module(ep)
    manifest=json.loads((args.prior/'manifest.json').read_text());config=manifest['config']
    inputs=json.loads((args.prior/'inputs.json').read_text())['pages']
    source=json.loads((args.prior/'source.json').read_text())['states']
    original=json.loads((args.prior/'protocol.json').read_text())
    assert len(inputs)==len(source)==verification['pages']
    for name,sha in manifest['artifacts_sha256'].items():
        assert ep.digest(name)==sha,name
    torch.manual_seed(0);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cuda.enable_cudnn_sdp(False)
    if torch.cuda.get_device_name(0)!='NVIDIA A100-SXM4-80GB':
        raise ValueError('Protocol requires the same A100 model')
    proc=ColQwen2_5_Processor.from_pretrained(config['processor_model'],local_files_only=True)
    proc.image_processor.max_pixels=original['max_pixels']
    proc.image_processor.size['longest_edge']=original['max_pixels']
    model,base,diagnostics=ep.load_model(config['targets']['vidore-v0.2'],config,torch)
    rows={n:pq.read_table(s['parquet'],columns=['image']).to_pylist() for n,s in config['collections'].items()}
    def native(i):
        item=inputs[i];value=rows[item['collection']][item['row']]['image']
        blob=value['bytes'] if isinstance(value,dict) else value
        assert hashlib.sha256(blob).hexdigest()==item['image_sha256']
        batch=proc.process_images([Image.open(io.BytesIO(blob)).convert('RGB')]).to('cuda:0')
        return model(**batch)[0][batch['attention_mask'][0].bool()].cpu()
    def replay(i):
        path=args.prior/'states'/source[i]['file']
        assert ep.digest(path)==source[i]['sha256']
        state=torch.load(path,weights_only=True,map_location='cpu')
        return ep.replay(base,state,torch).cpu()
    def timed(fn,i):
        torch.cuda.synchronize();start=time.perf_counter();value=fn(i)
        torch.cuda.synchronize();return value,time.perf_counter()-start
    def gpu_snapshot():
        return subprocess.run(['nvidia-smi','pmon','-c','1'],capture_output=True,text=True).stdout
    (args.output/'host-start.txt').write_text(gpu_snapshot())
    records=[]
    with torch.inference_mode():
        native(0);replay(0)
        for i,item in enumerate(inputs):
            if (i+args.round)%2:
                b,tb=timed(replay,i);a,ta=timed(native,i)
            else:
                a,ta=timed(native,i);b,tb=timed(replay,i)
            same=a.shape==b.shape and a.dtype==b.dtype
            record=dict(index=i,page=item,raw_seconds=ta,replay_seconds=tb,
                bit_equal=bool(same and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))),
                max_abs_error=float((a.float()-b.float()).abs().max()) if same else None,
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
            records.append(record)
            with (args.output/'pages.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if i%100==0:
                print(json.dumps({'round':args.round,'pages':i+1,'bit_equal':record['bit_equal']}),flush=True)
                (args.output/f'host-{i:04d}.txt').write_text(gpu_snapshot())
    raw=sum(r['raw_seconds'] for r in records);replay_seconds=sum(r['replay_seconds'] for r in records)
    result=dict(round=args.round,pid=os.getpid(),pages=len(records),raw_seconds=raw,replay_seconds=replay_seconds,
        ratio=replay_seconds/raw,unequal_pages=sum(not r['bit_equal'] for r in records),
        nonfinite_pages=sum(not r['finite'] for r in records),diagnostics=diagnostics,protocol=protocol,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        environment={name:version(name) for name in ['torch','transformers','peft','colpali-engine','pyarrow']},
        source_manifest_sha256=ep.digest(args.prior/'manifest.json'),verification_sha256=ep.digest(args.verification),
        code_sha256={p.name:ep.digest(p) for p in [Path(__file__),Path(__file__).with_name('run.py'),Path(__file__).with_name('repeat-protocol.json')]})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
