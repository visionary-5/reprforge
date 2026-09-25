#!/usr/bin/env python3
"""Aggregate sweep rows across families into one verdict table."""
import json, sys, glob
import os
from pathlib import Path
W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.path.expandvars("${INPUTS}/followup/item1"))
rows = []
for f in sorted(glob.glob(str(W / "out-*" / "rows.jsonl"))):
    fam = Path(f).parent.name[4:]
    src = json.loads((Path(f).parent / "source.json").read_text())
    for l in open(f):
        if l.strip():
            r = json.loads(l); r["family"] = fam; r["source_state_mib"] = src.get("state_mib_per_page_common"); rows.append(r)
# latest row per (family, target) wins (re-runs append)
latest = {}
for r in rows:
    latest[(r["family"], r["target"])] = r
rows = list(latest.values())
from collections import Counter
print("pairs:", len(rows)); print(Counter(r["verdict"] for r in rows))
print()
print(f"{'family':18s} {'target':52s} {'kind':8s} {'verdict':22s} native(acc/rej/bit)  common(acc/rej/bit)  reason")
for r in sorted(rows, key=lambda r: (r["family"], r["verdict"], r["target"])):
    n = r.get("native") or {}; c = r.get("common") or {}
    ns = f"{n.get('accepted','-')}/{n.get('rejected','-')}/{n.get('bitwise','-')}" if n else "-"
    cs = f"{c.get('accepted','-')}/{c.get('rejected','-')}/{c.get('bitwise','-')}" if c else "-"
    fc = r.get("forced_reuse_control") or {}
    extra = f" forced: {fc.get('bitwise')}/{fc.get('pages')} maxerr={fc.get('max_err', 0):.3f}" if fc else ""
    print(f"{r['family']:18s} {r['target'][:52]:52s} {r['kind']:8s} {r['verdict']:22s} {ns:20s} {cs:20s} {r.get('reason','')[:70]}{extra}")
json.dump(rows, open(W / "all_rows.json", "w"), indent=1, default=str)
