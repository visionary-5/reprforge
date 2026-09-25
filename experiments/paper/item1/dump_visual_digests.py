#!/usr/bin/env python3
"""Dump per-tensor digests of a base model's visual branch (parameters + persistent buffers as serialized) keyed by
name relative to the visual branch, using the same digest formula as the loaded-parameter check."""
import hashlib, json, sys
import torch
from safetensors import safe_open
from pathlib import Path
base_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
patterns = sys.argv[3].split(",") if len(sys.argv) > 3 else ["visual.", "vision_tower.", "multi_modal_projector."]
digests = {}
for f in sorted(base_dir.glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as sf:
        for name in sf.keys():
            rel = next((name[name.find(p):] for p in patterns if name.find(p) >= 0), None)
            if rel is None:
                continue
            t = sf.get_tensor(name)
            h = hashlib.sha256(); h.update(f"{list(t.shape)}:{t.dtype}".encode()); h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
            digests[rel] = h.hexdigest()
out.write_text(json.dumps(digests, indent=0))
print(len(digests), "visual tensors from", base_dir)
