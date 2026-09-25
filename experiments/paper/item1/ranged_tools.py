#!/usr/bin/env python3
"""Ranged-read tools over hf-mirror for releases whose full weights are too large to download.

  head <base_repo> <out.json>                    fetch custom_text_proj.{weight,bias} of a declared base
  visual-check <repo> <source_digests.json> <out.json> [--patterns p1,p2]
        parse every safetensors header of the release, range-download only the visual-branch tensors, digest each
        exactly as the loaded-parameter check does (sha256 over "shape:dtype" + raw bytes) and compare with the
        source's digests by name relative to the visual branch.
"""
import argparse, hashlib, json, os, struct, sys, time, urllib.request
from pathlib import Path
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_url, HfApi

DTYPE = {"BF16": "torch.bfloat16", "F16": "torch.float16", "F32": "torch.float32", "I64": "torch.int64", "I32": "torch.int32", "BOOL": "torch.bool", "U8": "torch.uint8", "F64": "torch.float64"}
DEFAULT_PATTERNS = ["visual.", "vision_tower.", "multi_modal_projector.", "vision_model.", "connector."]


def fetch(url, headers=None, retries=6):
    h = {"User-Agent": "Mozilla/5.0 (reprforge ranged)"}; h.update(headers or {})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def shard_header(repo, filename, revision):
    url = hf_hub_url(repo, filename, revision=revision)
    hlen = struct.unpack("<Q", fetch(url, {"Range": "bytes=0-7"}))[0]
    header = json.loads(fetch(url, {"Range": f"bytes=8-{8 + hlen - 1}"}))
    header.pop("__metadata__", None)
    return url, hlen, header


def shards_of(repo, revision):
    api = HfApi()
    files = [s.rfilename for s in api.model_info(repo, revision=revision).siblings]
    st = [f for f in files if f.endswith(".safetensors")]
    if not st:
        raise SystemExit(f"no safetensors in {repo}")
    return st


def relative_name(name, patterns):
    for p in patterns:
        i = name.find(p)
        if i >= 0:
            return name[i:]
    return None


def cmd_head(args):
    api = HfApi(); sha = api.model_info(args.repo).sha
    st = shards_of(args.repo, sha)
    out = {}
    shard_used = None
    for f in st:
        url, hlen, header = shard_header(args.repo, f, sha)
        for key in ("custom_text_proj.weight", "custom_text_proj.bias"):
            if key in header and key not in out:
                s, e = header[key]["data_offsets"]
                raw = fetch(url, {"Range": f"bytes={8 + hlen + s}-{8 + hlen + e - 1}"})
                out[key] = {"dtype": header[key]["dtype"], "shape": header[key]["shape"], "bytes": raw.hex()}
                shard_used = f
        if len(out) == 2:
            break
    if len(out) != 2:
        raise SystemExit(f"head not found in {args.repo}: {list(out)}")
    Path(args.out).write_text(json.dumps({"base": args.repo, "revision": sha, "shard": shard_used, "tensors": out}))
    print("head ok", args.repo, sha[:8], {k: (v["dtype"], v["shape"]) for k, v in out.items()})


def cmd_visual_check(args):
    patterns = args.patterns.split(",") if args.patterns else DEFAULT_PATTERNS
    source = json.loads(Path(args.source).read_text())  # relative name -> digest
    api = HfApi(); info = api.model_info(args.repo); sha = info.sha
    st = shards_of(args.repo, sha)
    result = {"repo": args.repo, "revision": sha, "shards": st, "tensors": {}, "bytes_downloaded": 0, "started": time.time()}
    for f in st:
        url, hlen, header = shard_header(args.repo, f, sha)
        for name, spec in header.items():
            rel = relative_name(name, patterns)
            if rel is None:
                continue
            s, e = spec["data_offsets"]
            raw = fetch(url, {"Range": f"bytes={8 + hlen + s}-{8 + hlen + e - 1}"})
            result["bytes_downloaded"] += len(raw)
            h = hashlib.sha256(); h.update(f"{spec['shape']}:{DTYPE.get(spec['dtype'], spec['dtype'])}".encode()); h.update(raw)
            result["tensors"][rel] = {"name": name, "dtype": spec["dtype"], "shape": spec["shape"], "digest": h.hexdigest(), "equal": source.get(rel) == h.hexdigest(), "in_source": rel in source}
            if len(result["tensors"]) % 50 == 0:
                print(json.dumps({"repo": args.repo, "tensors": len(result["tensors"]), "MB": result["bytes_downloaded"] / 1e6}), flush=True)
    names = set(result["tensors"])
    result.update({"visual_tensors": len(names), "equal": sum(t["equal"] for t in result["tensors"].values()),
                   "changed": sum((not t["equal"]) and t["in_source"] for t in result["tensors"].values()),
                   "not_in_source": sorted(n for n in names if not result["tensors"][n]["in_source"])[:20],
                   "missing_from_release": sorted(set(source) - names)[:20], "missing_count": len(set(source) - names),
                   "seconds": time.time() - result["started"]})
    result["verdict"] = "VISUAL_IDENTICAL" if result["changed"] == 0 and result["missing_count"] == 0 and not result["not_in_source"] else "VISUAL_DIFFERS"
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, default=str), flush=True)


ap = argparse.ArgumentParser(description=__doc__)
sub = ap.add_subparsers(dest="cmd", required=True)
h = sub.add_parser("head"); h.add_argument("repo"); h.add_argument("out"); h.set_defaults(fn=cmd_head)
v = sub.add_parser("visual-check"); v.add_argument("repo"); v.add_argument("source"); v.add_argument("out"); v.add_argument("--patterns"); v.set_defaults(fn=cmd_visual_check)
a = ap.parse_args(); a.fn(a)
