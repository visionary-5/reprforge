#!/usr/bin/env python3
"""End-to-end index rebuild at deployment scale (ColQwen2.5, common processing configuration).

Phases (each runs in its own process; orchestrate with run_e2e.sh):
  corpus            enumerate every unique page across the configured parquet collections,
                    fix a deterministic interleaved order, write corpus.json
  source-plain      source retriever builds an index without retention (baseline initial build)
  source-retain     source retriever builds an index AND retains the post-vision cut state per page
  target-full       target retriever rebuilds the index from raw pages (reference route)
  target-reprforge  target retriever rebuilds the index from retained states
                    (model-level check, per-page read + integrity check + resume, vector write, seal, publish)
  verify            bitwise comparison of the two target generations (not part of rebuild timing)

Every phase records wall-clock for the whole process (model load, loop, seal, publish) and per-page
stage timings. Cumulative loop time is logged at the requested page-count checkpoints so that the
scaling curve comes from one run rather than from separate runs.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("REPRFORGE_REPO", str(Path(__file__).resolve().parents[3])))
sys.path.insert(0, str(REPO))


def now():
    return time.perf_counter()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- corpus

def image_blob(value):
    if isinstance(value, dict):
        return value["bytes"]
    return value


def build_corpus(config, root, limit=None):
    import pyarrow.parquet as pq

    streams = []  # (collection, parquet path, image column)
    for coll in config["collections"]:
        paths = sorted(Path(x) for x in glob.glob(coll["parquet_glob"])) if "parquet_glob" in coll else [Path(coll["parquet"])]
        for p in paths:
            streams.append((coll["name"], str(p), coll["image_column"]))
    seen = {}
    per_stream = []  # list of list-of-rowgroups; each rowgroup -> list of (idx, sha, nbytes)
    duplicates = 0
    for name, path, col in streams:
        pf = pq.ParquetFile(path)
        groups = []
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=[col])
            rows = []
            for idx, value in enumerate(table.column(col).to_pylist()):
                blob = image_blob(value)
                if blob is None:
                    continue
                sha = sha256_bytes(blob)
                if sha in seen:
                    duplicates += 1
                    continue
                seen[sha] = (name, path, rg, idx)
                rows.append({"idx": idx, "sha": sha, "bytes": len(blob)})
            groups.append(rows)
            print(json.dumps({"corpus": name, "file": Path(path).name, "rg": rg, "unique_so_far": len(seen)}), flush=True)
        per_stream.append({"collection": name, "parquet": path, "image_column": col, "row_groups": groups})
    # Deterministic interleaving: round-robin over streams at row-group granularity.
    order = []
    cursors = [0] * len(per_stream)
    while True:
        progressed = False
        for s, stream in enumerate(per_stream):
            if cursors[s] < len(stream["row_groups"]):
                rg = cursors[s]
                cursors[s] += 1
                for row in stream["row_groups"][rg]:
                    order.append({"collection": stream["collection"], "parquet": stream["parquet"],
                                  "image_column": stream["image_column"], "rg": rg, **row})
                progressed = True
        if not progressed:
            break
    if limit:
        order = order[:limit]
    counts = {}
    for page in order:
        counts[page["collection"]] = counts.get(page["collection"], 0) + 1
    write_json(root / "corpus.json", {"pages": order, "unique_pages": len(order), "duplicates_dropped": duplicates,
                                      "per_collection": counts, "streams": [(s["collection"], s["parquet"]) for s in per_stream]})
    print(json.dumps({"corpus_done": len(order), "duplicates": duplicates, "per_collection": counts}), flush=True)


class PageReader:
    """Sequential reader over corpus order; holds one row group at a time."""

    def __init__(self, pages):
        import pyarrow.parquet as pq
        self.pq = pq
        self.pages = pages
        self.key = None
        self.column = None

    def blob(self, page):
        key = (page["parquet"], page["rg"])
        if key != self.key:
            pf = self.pq.ParquetFile(page["parquet"])
            self.column = pf.read_row_group(page["rg"], columns=[page["image_column"]]).column(page["image_column"]).to_pylist()
            self.key = key
        blob = image_blob(self.column[page["idx"]])
        if sha256_bytes(blob) != page["sha"]:
            raise RuntimeError(f"page bytes changed: {page['sha']}")
        return blob


# ----------------------------------------------------------------------------- storage

class ShardWriter:
    """Append-only BF16 vector shards with fsync per shard; offsets written at close."""

    def __init__(self, directory, pages_per_shard=1000):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.pages_per_shard = pages_per_shard
        self.offsets = []
        self.shard = -1
        self.handle = None
        self.position = 0
        self.count_in_shard = 0
        self.artifacts = []

    def _rotate(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        self.shard += 1
        name = f"vectors-{self.shard:04d}.bin"
        self.handle = (self.dir / name).open("wb")
        self.artifacts.append(name)
        self.position = 0
        self.count_in_shard = 0

    def append(self, page_id, tensor):
        if self.handle is None or self.count_in_shard >= self.pages_per_shard:
            self._rotate()
        import torch
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        self.handle.write(raw)
        self.offsets.append([page_id, self.shard, self.position, int(tensor.shape[0]), int(tensor.shape[1])])
        self.position += len(raw)
        self.count_in_shard += 1

    def close(self, dtype_name):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        with (self.dir / "offsets.json").open("w") as f:
            json.dump({"dtype": dtype_name, "records": self.offsets}, f)
            f.flush()
            os.fsync(f.fileno())
        self.artifacts.append("offsets.json")
        return list(self.artifacts)


def read_generation_vectors(gen_dir):
    """Yield (page_id, bf16 tensor) from a sealed generation."""
    import torch
    meta = json.loads((Path(gen_dir) / "offsets.json").read_text())
    handles = {}
    for page_id, shard, pos, n, d in meta["records"]:
        if shard not in handles:
            handles[shard] = (Path(gen_dir) / f"vectors-{shard:04d}.bin").open("rb")
        h = handles[shard]
        h.seek(pos)
        raw = h.read(n * d * 2)
        yield page_id, torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(n, d)


def state_path(root, page_id):
    return root / "states" / page_id[:2] / f"{page_id}.pt"


def evict_page_cache(paths):
    """Best-effort eviction of file pages from the OS page cache (no root required)."""
    evicted = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                evicted += 1
            finally:
                os.close(fd)
        except OSError:
            pass
    return evicted


# ----------------------------------------------------------------------------- model glue

def setup(config, args):
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from reprforge.integrations.colqwen import load_model

    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = ColQwen2_5_Processor.from_pretrained(config["processor_model"], local_files_only=True)
    processor.image_processor.max_pixels = config["max_pixels"]
    processor.image_processor.size["longest_edge"] = config["max_pixels"]
    spec = config["source"] if args.phase.startswith("source") else config["targets"][args.target]
    t0 = now()
    model, base, diagnostics = load_model(spec, config, torch)
    load_seconds = now() - t0
    return torch, processor, model, base, diagnostics, load_seconds


def contract_of(base, processor):
    spec = importlib.util.spec_from_file_location("rf_common", REPO / "scripts" / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.contract(base, processor)


def environment():
    import torch
    return {"packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "colpali-engine", "numpy", "pyarrow", "safetensors", "pillow")},
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), "threads": torch.get_num_threads()}


def timed(torch, fn):
    torch.cuda.synchronize()
    t = now()
    out = fn()
    torch.cuda.synchronize()
    return out, now() - t


# ----------------------------------------------------------------------------- phases

def run_build(config, root, args):
    """source-plain | source-retain | target-full | target-reprforge."""
    import torch
    from PIL import Image
    from reprforge.generation import publish_generation, seal_generation
    from reprforge.integrations.colqwen import capture, replay

    phase = args.phase
    gen_name = args.generation or (phase if phase.startswith("source") else f"{phase}-{args.target}")
    corpus = json.loads((root / "corpus.json").read_text())
    pages = corpus["pages"][: args.limit] if args.limit else corpus["pages"]
    checkpoints = set(int(x) for x in args.checkpoints.split(",") if x)
    process_start = now()
    torch_, processor, model, base, diagnostics, load_seconds = setup(config, args)
    reader = PageReader(pages)
    gen_dir = root / "generations" / gen_name
    if gen_dir.exists():
        raise RuntimeError(f"generation exists: {gen_dir}")
    rows_path = root / f"{gen_name}-pages.jsonl"
    if rows_path.exists():
        raise RuntimeError(f"rows exist: {rows_path}")
    rows_handle = rows_path.open("w")

    contract = None
    validation = {}
    source_manifest = None
    if phase == "target-reprforge":
        # Model-level check (once per transition): compare the target's visual-path contract with the source's.
        source_manifest = json.loads((root / "source-retain.json").read_text())
        t = now()
        contract = json.loads(json.dumps(contract_of(base, processor), default=str))
        keys = ("vision", "processor", "model_config", "dtype")
        valid = all(contract[k] == source_manifest["contract"][k] for k in keys)
        validation = {"seconds": now() - t, "valid": valid,
                      "compared": {k: contract[k] == source_manifest["contract"][k] for k in contract if k in source_manifest["contract"]}}
        if not valid:
            raise RuntimeError(f"model-level check failed: {validation}")
        state_index = {s["page"]: s for s in source_manifest["states"]}
        # Evict retained states from the page cache so reads are cold; this is preparation, not rebuild time.
        t = now()
        evicted = evict_page_cache([state_path(root, p["sha"]) for p in pages])
        validation["page_cache_evicted_files"] = evicted
        validation["page_cache_evict_seconds"] = now() - t
    elif phase == "source-retain":
        contract = json.loads(json.dumps(contract_of(base, processor), default=str))
        for p in pages:
            state_path(root, p["sha"]).parent.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(gen_dir)
    state_records = []
    cum = {}
    stage_totals = {}
    loop_start = None

    def add(rec):
        for k, v in rec.items():
            if k.startswith("t_"):
                stage_totals[k] = stage_totals.get(k, 0.0) + v

    def decode(page):
        blob = reader.blob(page)
        return Image.open(io.BytesIO(blob)).convert("RGB")

    # Warm-up on the first page (no rows recorded, no vectors written) so kernel/cuDNN autotuning does not land on page 0.
    with torch.inference_mode():
        warm = processor.process_images([decode(pages[0])]).to("cuda:0")
        if phase == "target-reprforge":
            st = torch.load(state_path(root, pages[0]["sha"]), map_location="cpu", weights_only=True)
            replay(base, {k: v for k, v in st.items() if k in ("input_ids", "attention_mask", "image_grid_thw", "vision")}, torch)
        else:
            model(**warm)
        del warm
        torch.cuda.synchronize()
        loop_start = now()
        for i, page in enumerate(pages):
            pid = page["sha"]
            rec = {"i": i, "page": pid, "collection": page["collection"]}
            if phase in ("source-plain", "source-retain", "target-full"):
                t = now(); image = decode(page); rec["t_decode"] = now() - t
                t = now(); batch = processor.process_images([image]); rec["t_process"] = now() - t
                t = now(); batch = batch.to("cuda:0"); torch.cuda.synchronize(); rec["t_h2d"] = now() - t
                if phase == "source-retain":
                    state, rec["t_vision"] = timed(torch, lambda: capture(base, batch, torch))
                    state["processor_contract"] = contract["processor"]
                    vectors, rec["t_resume"] = timed(torch, lambda: replay(base, {k: state[k] for k in ("input_ids", "attention_mask", "image_grid_thw", "vision")}, torch))
                    t = now()
                    buf = io.BytesIO(); torch.save(state, buf); raw = buf.getvalue(); digest = sha256_bytes(raw)
                    rec["t_state_serialize"] = now() - t
                    t = now()
                    sp = state_path(root, pid)
                    with sp.open("wb") as f:
                        f.write(raw); f.flush(); os.fsync(f.fileno())
                    rec["t_state_write"] = now() - t
                    rec["state_bytes"] = len(raw)
                    state_records.append({"page": pid, "sha256": digest, "bytes": len(raw), "tokens": int(state["vision"].shape[0])})
                else:
                    out, rec["t_forward"] = timed(torch, lambda _model=model: _model(**batch))
                    vectors = out[0][batch["attention_mask"][0].bool()]
                t = now(); vectors = vectors.detach().to("cpu"); torch.cuda.synchronize(); writer.append(pid, vectors); rec["t_vector_write"] = now() - t
            else:  # target-reprforge
                sp = state_path(root, pid)
                t = now()
                with sp.open("rb") as f:
                    raw = f.read()
                rec["t_state_read"] = now() - t
                t = now()
                expected = state_index[pid]
                if sha256_bytes(raw) != expected["sha256"]:
                    raise RuntimeError(f"state integrity failure: {pid}")
                rec["t_state_verify"] = now() - t
                t = now(); state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True); rec["t_state_deserialize"] = now() - t
                t = now()
                if state.get("processor_contract") != contract["processor"]:
                    raise RuntimeError(f"page-level processing contract mismatch: {pid}")
                rec["t_page_check"] = now() - t
                vectors, rec["t_resume"] = timed(torch, lambda: replay(base, {k: state[k] for k in ("input_ids", "attention_mask", "image_grid_thw", "vision")}, torch))
                t = now(); vectors = vectors.detach().to("cpu"); torch.cuda.synchronize(); writer.append(pid, vectors); rec["t_vector_write"] = now() - t
            rec["tokens"] = int(vectors.shape[0])
            rec["finite"] = bool(torch.isfinite(vectors).all())
            add(rec)
            rows_handle.write(json.dumps(rec) + "\n")
            n = i + 1
            if n in checkpoints or n == len(pages):
                cum[str(n)] = now() - loop_start
                print(json.dumps({"phase": phase, "generation": gen_name, "pages": n, "loop_seconds": cum[str(n)],
                                  "per_page": cum[str(n)] / n}), flush=True)
            elif n % 200 == 0:
                print(json.dumps({"phase": phase, "pages": n, "loop_seconds": now() - loop_start}), flush=True)
    loop_seconds = now() - loop_start
    rows_handle.close()
    t = now()
    artifacts = writer.close("bfloat16")
    write_json(gen_dir / "layout.json", {"format": "bf16 token matrices, row-major, per-page contiguous", "dimension": 128, "pages": len(pages), "route": phase, "target": args.target})
    artifacts.append("layout.json")
    seal_generation(root, gen_name, artifacts)
    publish_generation(root, gen_name)
    publish_seconds = now() - t
    total_seconds = now() - process_start
    summary = {"phase": phase, "generation": gen_name, "target": args.target, "pages": len(pages), "max_pixels": config["max_pixels"],
               "model_load_seconds": load_seconds, "validation": validation, "loop_seconds": loop_seconds,
               "seal_publish_seconds": publish_seconds, "process_wall_seconds": total_seconds,
               "cumulative_loop_seconds_at": cum, "stage_totals_seconds": stage_totals,
               "vector_bytes": sum((gen_dir / a).stat().st_size for a in artifacts if a.startswith("vectors-")),
               "diagnostics": {k: v for k, v in diagnostics.items() if k != "unexpected"}, "environment": environment(),
               "contract": contract, "batch_size": 1, "generation_dir": str(gen_dir)}
    if phase == "source-retain":
        summary["states"] = state_records
        summary["state_bytes_total"] = sum(s["bytes"] for s in state_records)
    write_json(root / f"{gen_name}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("states", "contract", "diagnostics", "environment")}), flush=True)


def run_verify(config, root, args):
    import torch
    from reprforge.generation import validate_generation
    a_name, b_name = f"target-full-{args.target}", f"target-reprforge-{args.target}"
    for name in (a_name, b_name):
        validate_generation(root, name)
    a = read_generation_vectors(root / "generations" / a_name)
    b = read_generation_vectors(root / "generations" / b_name)
    pages = equal = 0
    elements = equal_elements = 0
    max_err = 0.0
    mismatched = []
    for (pa, va), (pb, vb) in zip(a, b):
        if pa != pb:
            raise RuntimeError("page order differs between generations")
        pages += 1
        same_shape = va.shape == vb.shape
        elements += va.numel()
        if same_shape:
            eq = torch.equal(va.view(torch.uint8), vb.view(torch.uint8))
            equal += int(eq)
            equal_elements += int((va == vb).sum())
            err = float((va.float() - vb.float()).abs().max())
            max_err = max(max_err, err)
            if not eq:
                mismatched.append({"page": pa, "max_abs_error": err})
        else:
            mismatched.append({"page": pa, "shape": [list(va.shape), list(vb.shape)]})
    result = {"target": args.target, "pages": pages, "bitwise_equal_pages": equal, "elements": elements, "equal_elements": equal_elements,
              "max_abs_error": max_err, "mismatched": mismatched[:50], "mismatched_count": len(mismatched)}
    write_json(root / f"verify-{args.target}.json", result)
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=["corpus", "source-plain", "source-retain", "target-full", "target-reprforge", "verify"])
    ap.add_argument("--target")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--generation")
    ap.add_argument("--checkpoints", default="1000,3000,10000,20000")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase == "corpus":
        build_corpus(config, args.root, args.limit)
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES required")
    if args.phase == "verify":
        run_verify(config, args.root, args)
    else:
        if args.phase.startswith("target") and not args.target:
            raise SystemExit("--target required")
        run_build(config, args.root, args)


if __name__ == "__main__":
    main()
