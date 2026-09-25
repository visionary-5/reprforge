#!/usr/bin/env python3
"""Collect the interface-trace results across architectures into one table."""
import json, sys
import os
from pathlib import Path
W = Path(os.path.expandvars("${INPUTS}/followup/item3"))
rows = []
def frontier_desc(fr):
    return [f"{tuple(f['shape'])} <- {f['source_module']}[{f.get('source_leaf_index')}] -> {f['consumer_module'] or '<root>'}:{f['consumer_op'].replace('aten.','')}" for f in fr]
# ColQwen2.5
s = json.load(open(W / "out-colqwen25/summary.json")); t = s["trace"]
rows.append({"architecture": "Qwen2.5-VL-3B (ColQwen2.5)", "trace_s": t["trace_seconds"], "ops": t["op_count"], "frontier": frontier_desc(t["frontier"]), "cut_modules": t["cut_modules"],
             "pre_cut": t["pre_cut_params"], "post_cut": t["post_cut_params"], "structural": t["structural_params"], "hand_written": t["hand_written_visual_state_dict"], "traced_minus_hand": t["traced_minus_hand"],
             "targets": {k: (v["generic_bitwise"], v["pages"], v["model_level_ok"]) for k, v in s["targets"].items()}})
s = json.load(open(W / "out-qwen3-2b/summary.json")); t = s["trace"]; r = s["rudore"]
rows.append({"architecture": "Qwen3-VL-2B (SauerkrautLM-ColQwen3, DeepStack)", "trace_s": t["trace_seconds"], "ops": t["op_count"], "frontier": frontier_desc(t["frontier"]), "cut_modules": t["cut_modules"],
             "pre_cut": t["pre_cut_params"], "post_cut": t["post_cut_params"], "structural": t["structural_params"], "hand_written": t["hand_visual_prefix_params"], "traced_minus_hand": t["traced_minus_hand"],
             "targets": {"+RuDoRe": (r["generic_bitwise"], r["pages"], r["model_level_ok"])}, "state_mib": r["state_mib_per_page"], "control_main_only": f"{r['control_main_tokens_only_bitwise']}/{r['control_pages']}"})
s = json.load(open(W / "out-qwen3-4b/summary.json")); r = s["result"]
rows.append({"architecture": "Qwen3-VL-4B, two publisher wrappers (Tomoro -> OpenSearch)", "trace_s": r["trace_seconds"], "frontier": [str(f) for f in s["source"]["frontier"]], "cut_modules": s["source"]["cut_modules"], "target_cut_modules": s["target"]["cut_modules"],
             "pre_cut": s["source"]["pre_cut"], "post_cut": s["source"]["post_cut"], "targets": {"OpenSearch 4B": (r["generic_bitwise"], r["pages"], r["model_level_ok"])}, "state_mib": r["state_mib_per_page"], "module_mapping": r["module_mapping"]})
for name, label in [("out-colpali-v10", "PaliGemma-3B (ColPali v1.1 -> v1.0)")]:
    s = json.load(open(W / f"{name}/summary.json")); t = s["trace"]
    rows.append({"architecture": label, "trace_s": t["trace_seconds"], "ops": t["op_count"], "frontier": frontier_desc(t["frontier"]), "cut_modules": t["cut_modules"], "visual_modules": t["visual_modules"],
                 "pre_cut": t["pre_cut_params"], "post_cut": t["post_cut_params"], "targets": {k: (v["generic_bitwise"], v["pages"], v["model_level_ok"]) for k, v in s["targets"].items() if k != "state_mib_per_page"}, "state_mib": s["targets"]["state_mib_per_page"]})
json.dump(rows, open(W / "item3_table.json", "w"), indent=1, default=str)
for r in rows:
    print("==", r["architecture"]); [print("  ", k, ":", v) for k, v in r.items() if k != "architecture"]
