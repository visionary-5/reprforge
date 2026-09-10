"""Independent target endpoint for the frozen stratified page-validity check."""

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reprforge.equivalence import fingerprint_component_outputs
from reprforge.page_reuse import assess_page_reuse
from reprforge.versions import VersionManifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from PIL import Image

    script = Path(__file__).parents[1] / "independent-endpoint" / "run.py"
    spec = importlib.util.spec_from_file_location("independent_endpoint", script)
    endpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(endpoint)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    manifest = json.loads((args.prior / "manifest.json").read_text())
    config = manifest["config"]
    # Reverify the old pinned artifacts, rather than trust path or release names.
    for name, expected in manifest["artifacts_sha256"].items():
        if endpoint.digest(Path(name)) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    records = [
        json.loads(line)
        for line in (args.probe / "pages.jsonl").read_text().splitlines()
    ]
    selected = [
        r
        for equal in (True, False)
        for r in [x for x in records if x["equal"] == equal][:8]
    ]
    if not any(r["equal"] for r in selected) or all(r["equal"] for r in selected):
        raise ValueError("CPU gate did not admit a mixed sample")
    source_states = json.loads((args.prior / "source.json").read_text())["states"]
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    processors = []
    for budget in (protocol["source_max_pixels"], protocol["target_max_pixels"]):
        proc = ColQwen2_5_Processor.from_pretrained(
            config["processor_model"], local_files_only=True
        )
        proc.image_processor.max_pixels = budget
        proc.image_processor.size["longest_edge"] = budget
        processors.append(proc)
    rows = {
        n: pq.read_table(s["parquet"], columns=["image"]).to_pylist()
        for n, s in config["collections"].items()
    }
    model, base, diagnostics = endpoint.load_model(
        config["targets"]["vidore-v0.2"], config, torch
    )
    upstream = hashlib.sha256(
        json.dumps(manifest["artifacts_sha256"], sort_keys=True).encode()
    ).hexdigest()

    def version(budget, adapter):
        return VersionManifest(
            "same-frozen-pages",
            f"{upstream}:budget={budget}",
            upstream,
            upstream,
            adapter,
            "target-head",
            "unchanged",
        )

    source = version(protocol["source_max_pixels"], "source")
    target = version(protocol["target_max_pixels"], "target")
    results, banks = [], []

    def bit_equal(a, b):
        return (
            a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
        )

    with torch.inference_mode():
        for item in selected:
            value = rows[item["collection"]][item["row"]]["image"]
            blob = value["bytes"] if isinstance(value, dict) else value
            assert hashlib.sha256(blob).hexdigest() == item["image_sha256"]

            def processed(which):
                return processors[which].process_images(
                    [Image.open(io.BytesIO(blob)).convert("RGB")]
                )

            batches = [processed(0), processed(1)]
            fingerprints = [
                fingerprint_component_outputs(
                    scope_fingerprint=item["image_sha256"],
                    outputs=[{k: v.cpu().numpy() for k, v in batch.items()}],
                )
                for batch in batches
            ]
            decision = assess_page_reuse(
                source,
                target,
                page_fingerprint=item["image_sha256"],
                dependencies=frozenset({"source", "processor", "vision"}),
                required_processor_fields=frozenset(
                    {"pixel_values", "input_ids", "attention_mask", "image_grid_thw"}
                ),
                source_outputs=fingerprints[0],
                target_outputs=fingerprints[1],
            )
            assert decision.replay == item["equal"]
            state_record = source_states[item["index"]]
            state_path = args.prior / "states" / state_record["file"]
            assert endpoint.digest(state_path) == state_record["sha256"]
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            for key in ("input_ids", "attention_mask", "image_grid_thw"):
                assert bit_equal(state[key], batches[0][key])

            def native():
                batch = processed(1).to("cuda:0")
                return model(**batch)[0][batch["attention_mask"][0].bool()].cpu()

            raw = native()
            unsafe = endpoint.replay(base, state, torch).cpu()
            # Execute fallback independently; do not copy the reference bank.
            routed = unsafe if decision.replay else native()
            rec = {
                "index": item["index"],
                "collection": item["collection"],
                "image_sha256": item["image_sha256"],
                "state_sha256": state_record["sha256"],
                "route": "replay" if decision.replay else "raw",
                "blockers": sorted(decision.blockers),
                "source_input": fingerprints[0].to_dict(),
                "target_input": fingerprints[1].to_dict(),
                "raw_shape": list(raw.shape),
                "unsafe_shape": list(unsafe.shape),
                "finite": bool(
                    torch.isfinite(raw).all() and torch.isfinite(routed).all()
                ),
                "routed_bit_equal": bit_equal(raw, routed),
                "unsafe_bit_equal": bit_equal(raw, unsafe),
            }
            results.append(rec)
            banks.append(
                {"index": item["index"], "raw": raw, "routed": routed, "unsafe": unsafe}
            )
            print(json.dumps(rec), flush=True)
    torch.save(banks, args.output / "banks.pt")
    summary = {
        "pages": len(results),
        "replay_pages": sum(r["route"] == "replay" for r in results),
        "routed_bit_equal": sum(r["routed_bit_equal"] for r in results),
        "unsafe_bit_equal": sum(r["unsafe_bit_equal"] for r in results),
        "finite_pages": sum(r["finite"] for r in results),
        "diagnostics": diagnostics,
        "protocol": protocol,
        "pages_detail": results,
        "banks_sha256": endpoint.digest(args.output / "banks.pt"),
        "code_sha256": {
            str(p.relative_to(Path(__file__).parents[2])): endpoint.digest(p)
            for p in (
                Path(__file__),
                script,
                Path(__file__).parents[2] / "reprforge" / "page_reuse.py",
            )
        },
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "BF16",
            "batch": 1,
            "cudnn_sdpa": False,
            "tf32": True,
        },
        "scope": "Stratified diagnostic only; no population fidelity or speed estimate. Processor fingerprints reconstructed retrospectively, old state hashes verified.",
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
