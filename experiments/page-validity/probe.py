"""CPU-only diagnostic; preserve the frozen input order and raw image bytes."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path


def fingerprint(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        tensor = batch[key].contiguous().cpu()
        meta = json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode()
        payload = tensor.numpy().tobytes()
        h.update(len(meta).to_bytes(8, "little") + meta)
        h.update(len(payload).to_bytes(8, "little") + payload)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from PIL import Image

    torch.set_num_threads(4)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    config = json.loads((args.prior / "manifest.json").read_text())["config"]
    inputs = json.loads((args.prior / "inputs.json").read_text())
    processors = []
    for budget in (protocol["source_max_pixels"], protocol["target_max_pixels"]):
        proc = ColQwen2_5_Processor.from_pretrained(
            config["processor_model"], local_files_only=True)
        proc.image_processor.max_pixels = budget
        proc.image_processor.size["longest_edge"] = budget
        processors.append(proc)
    rows = {n: pq.read_table(s["parquet"], columns=["image"]).to_pylist()
            for n, s in config["collections"].items()}
    records = []
    for index, item in enumerate(inputs["pages"]):
        value = rows[item["collection"]][item["row"]]["image"]
        blob = value["bytes"] if isinstance(value, dict) else value
        assert hashlib.sha256(blob).hexdigest() == item["image_sha256"]
        outputs, clocks, hashes = [], [], []
        for proc in processors:
            start = time.perf_counter()
            batch = proc.process_images([Image.open(io.BytesIO(blob)).convert("RGB")])
            pre = time.perf_counter() - start
            start = time.perf_counter()
            hashes.append(fingerprint(batch))
            clocks.append({"preprocess": pre, "fingerprint": time.perf_counter() - start})
            outputs.append(batch)
        changed = [key for key in set(outputs[0]) | set(outputs[1])
                   if key not in outputs[0] or key not in outputs[1]
                   or outputs[0][key].shape != outputs[1][key].shape
                   or outputs[0][key].dtype != outputs[1][key].dtype
                   or not torch.equal(outputs[0][key].contiguous().view(torch.uint8),
                                      outputs[1][key].contiguous().view(torch.uint8))]
        rec = dict(item, index=index, equal=not changed, changed_fields=sorted(changed),
                   fingerprints=hashes, clocks=clocks,
                   grids=[b["image_grid_thw"].tolist() for b in outputs])
        assert (hashes[0] == hashes[1]) == rec["equal"]
        records.append(rec)
        with (args.output / "pages.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if (index + 1) % 20 == 0:
            print(json.dumps({"pages": index + 1, "equal": sum(r["equal"] for r in records)}), flush=True)
    summary = {"pages": len(records), "equal": sum(r["equal"] for r in records),
               "by_collection": {n: {"pages": sum(r["collection"] == n for r in records),
                   "equal": sum(r["collection"] == n and r["equal"] for r in records)} for n in rows},
               "protocol": protocol,
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
