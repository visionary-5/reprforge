"""Full HR split native/common quality and matched validation timing on A100."""

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5_Processor

from reprforge import hooked
from reprforge.integrations.colqwen import load_model
from reprforge.tracing import discover_interface


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def write_tensor(path, tensor):
    with path.open("wb") as f:
        torch.save(tensor, f)
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing-pages", type=int, default=200)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name in ["states", "native", "common", "replay", "timing"]:
        (root / name).mkdir()
    dump(root / "config.json", cfg)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    native = ColQwen2_5_Processor.from_pretrained(
        cfg["native_processor"], local_files_only=True
    )
    # Match the manuscript's common base processor, not merely its pixel budget.
    common = ColQwen2_5_Processor.from_pretrained(
        cfg["processor_model"], local_files_only=True
    )
    common.image_processor.max_pixels = 602112
    common.image_processor.size["longest_edge"] = 602112
    recipe = digest(
        json.dumps(
            {
                "processor": common.to_dict(),
                "image": common.image_processor.to_dict(),
                "tokenizer": common.tokenizer.backend_tokenizer.to_str(),
            },
            sort_keys=True,
            default=str,
        ).encode()
    )
    dump(
        root / "processors.json",
        {
            "native": native.image_processor.to_dict(),
            "common": common.image_processor.to_dict(),
            "common_recipe_sha256": recipe,
        },
    )
    rows = sorted(
        pq.read_table(cfg["hr_corpus"]).to_pylist(), key=lambda r: r["corpus_id"]
    )
    if args.limit:
        rows = rows[: args.limit]
    dump(
        root / "pages.json",
        [{"id": r["corpus_id"], "sha256": digest(r["image"]["bytes"])} for r in rows],
    )

    def preprocess(row, proc):
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return proc.process_images([img])

    def gpu(b):
        return {k: v.cuda() for k, v in b.items()}

    def vector(model, b):
        return model(**b)[0][b["attention_mask"][0].bool()].detach().cpu()

    model, base, diag = load_model(cfg["source"], cfg, torch)
    b = gpu(preprocess(rows[0], common))
    with torch.inference_mode():
        model(**b)
        start = tick()
        report = discover_interface(
            base,
            lambda: model(**b),
            visual=[b["pixel_values"]],
            text=[b["input_ids"]],
            structural=[b["image_grid_thw"], b["attention_mask"]],
        )
        trace_seconds = tick() - start
        spec = hooked.CutSpec.from_report(report, list(report.execution_order))
        source_deps = hooked.dependency_digest(base, list(report.pre_cut_params))
        state_manifest = []
        for i, row in enumerate(rows):
            b = gpu(preprocess(row, common))
            st = hooked.capture(base, spec, lambda: model(**b))
            st.pop("result")
            st["inputs"] = {k: v.cpu() for k, v in b.items() if k != "pixel_values"}
            st["pixel_shape"] = list(b["pixel_values"].shape)
            st["pixel_dtype"] = str(b["pixel_values"].dtype).replace("torch.", "")
            st["recipe"] = recipe
            st["image_sha256"] = digest(row["image"]["bytes"])
            path = root / "states" / f"{i}.pt"
            write_tensor(path, st)
            state_manifest.append(
                {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
            )
            if i % 50 == 0:
                print("retain", i, flush=True)
    dump(root / "state_manifest.json", state_manifest)
    dump(root / "interface.json", report.to_dict())
    del model, base, b
    torch.cuda.empty_cache()
    start = tick()
    model, base, diag = load_model(cfg["targets"]["colnomic-3b"], cfg, torch)
    load_seconds = tick() - start
    start = tick()
    target_deps = hooked.dependency_digest(base, list(report.pre_cut_params))
    check_seconds = tick() - start
    assert target_deps == source_deps
    dump(
        root / "dependencies.json",
        {"source": source_deps, "target": target_deps, "equal": True},
    )

    def read_state(i):
        data = (root / "states" / f"{i}.pt").read_bytes()
        assert digest(data) == state_manifest[i]["sha256"]
        st = torch.load(io.BytesIO(data), weights_only=False)
        assert st["recipe"] == recipe
        assert st["image_sha256"] == digest(rows[i]["image"]["bytes"])
        return st

    raw = []
    with torch.inference_mode():
        for proc in [native, common]:
            model(**gpu(preprocess(rows[0], proc)))
        for i, row in enumerate(rows):
            rec = {"id": row["corpus_id"], "i": i}
            for name, proc in [("native", native), ("common", common)]:
                start = tick()
                b = gpu(preprocess(row, proc))
                v = vector(model, b)
                assert torch.isfinite(v).all()
                write_tensor(root / name / f"{i}.pt", v)
                rec[name + "_seconds"] = tick() - start
                rec[name + "_visual_tokens"] = int(
                    b["image_grid_thw"].prod().item() // 4
                )
                rec[name + "_pixels"] = int(b["image_grid_thw"].prod().item() * 14 * 14)
                if name == "common":
                    start = tick()
                    b = gpu(preprocess(row, common))
                    st = read_state(i)
                    replay = hooked.resume(base, st, lambda: vector(model, b))
                    write_tensor(root / "replay" / f"{i}.pt", replay)
                    rec["replay_seconds"] = tick() - start
                    rec["exact"] = bool(
                        torch.equal(v.view(torch.uint8), replay.view(torch.uint8))
                    )
                    rec["max_error"] = float((v.float() - replay.float()).abs().max())
                    assert rec["exact"] and torch.isfinite(v).all()
            raw.append(rec)
            with (root / "encoding.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print("encode", i, flush=True)

        # Three paired passes, route order rotated by pass and reversed on odd pages.
        timing = []
        indices = np.linspace(
            0, len(rows) - 1, min(args.timing_pages, len(rows)), dtype=int
        ).tolist()
        for rep in range(3):
            for i in indices:
                order = ["full", "generic", "fixed"]
                order = order[rep:] + order[:rep]
                if i % 2:
                    order.reverse()
                for route in order:
                    # Explicitly warm state-file cache for BOTH replay paths, outside timing.
                    (root / "states" / f"{i}.pt").read_bytes()
                    start = tick()
                    t = start
                    if route != "fixed":
                        b_cpu = preprocess(rows[i], common)
                    else:
                        b_cpu = None
                    t1 = tick()
                    st = read_state(i) if route != "full" else None
                    t2 = tick()
                    if route == "fixed":
                        b_cpu = dict(st["inputs"])
                        b_cpu["pixel_values"] = torch.empty(
                            st["pixel_shape"], dtype=getattr(torch, st["pixel_dtype"])
                        )
                    b = gpu(b_cpu)
                    t3 = tick()
                    hash_time = [0.0]
                    original = hooked.tensor_digest

                    def measured_hash(tensors):
                        h0 = tick()
                        value = original(tensors)
                        hash_time[0] += tick() - h0
                        return value

                    hooked.tensor_digest = measured_hash
                    try:
                        v = (
                            vector(model, b)
                            if route == "full"
                            else hooked.resume(
                                base,
                                st,
                                lambda: vector(model, b),
                                check_inputs=route == "generic",
                            )
                        )
                    finally:
                        hooked.tensor_digest = original
                    t4 = tick()
                    write_tensor(root / "timing" / f"{route}.pt", v)
                    end = tick()
                    ref = torch.load(root / "common" / f"{i}.pt", weights_only=True)
                    exact = bool(
                        torch.equal(v.view(torch.uint8), ref.view(torch.uint8))
                    )
                    assert exact
                    rec = {
                        "repeat": rep,
                        "i": i,
                        "route": route,
                        "order": order,
                        "total": end - start,
                        "decode_preprocess": t1 - t,
                        "read_validate": t2 - t1,
                        "h2d_and_placeholders": t3 - t2,
                        "input_hash": hash_time[0],
                        "compute_excluding_hash": t4 - t3 - hash_time[0],
                        "write": end - t4,
                        "exact": exact,
                    }
                    timing.append(rec)
                    with (root / "timing.jsonl").open("a") as f:
                        f.write(json.dumps(rec) + "\n")
            print("timing repeat", rep, "done", flush=True)

        # Full standard query set; each configuration uses its own query processor.
        data = Path(cfg["hr_corpus"]).parents[1]
        queries = pq.read_table(
            data / "queries/test-00000-of-00001.parquet"
        ).to_pylist()
        qrels_rows = pq.read_table(
            data / "qrels/test-00000-of-00001.parquet"
        ).to_pylist()
        qrels = {}
        for r in qrels_rows:
            qrels.setdefault(str(r["query_id"]), {})[str(r["corpus_id"])] = int(
                r["score"]
            )
        queries = sorted(queries, key=lambda r: str(r["query_id"]))
        assert len({str(q["query_id"]) for q in queries}) == len(queries)
        assert len({str(r["corpus_id"]) for r in rows}) == len(rows)
        if not args.limit:
            corpus_ids = {str(r["corpus_id"]) for r in rows}
            assert set(qrels) == {str(q["query_id"]) for q in queries}
            assert all(set(rel) <= corpus_ids for rel in qrels.values())
        if args.limit:
            queries = queries[:8]
        dump(
            root / "queries.json",
            [{"id": str(q["query_id"]), "query": q["query"]} for q in queries],
        )
        dump(root / "qrels.json", qrels)
        from experiments.support.streaming_maxsim import streaming_maxsim

        metrics = {}
        for name, proc in [("native", native), ("common", common)]:
            bank = []
            for q in queries:
                qb = gpu(proc.process_queries([q["query"]]))
                bank.append(vector(model, qb))
            torch.save(bank, root / f"{name}-queries.pt")
            docs = [
                torch.load(root / name / f"{i}.pt", weights_only=True)
                for i in range(len(rows))
            ]
            scores = streaming_maxsim(bank, docs, query_chunk=32, document_chunk=16)
            np.save(root / f"{name}-scores.npy", scores)
            ranks = np.argsort(-scores, axis=1, kind="stable")
            per_query = []
            for qi, q in enumerate(queries):
                rel = qrels.get(str(q["query_id"]), {})
                row_metric = {"id": str(q["query_id"])}
                for k in [5, 10]:
                    gains = np.array(
                        [rel.get(str(rows[j]["corpus_id"]), 0) for j in ranks[qi, :k]],
                        dtype=float,
                    )
                    ideal = sorted(rel.values(), reverse=True)[:k]
                    dcg = float((gains / np.log2(np.arange(len(gains)) + 2)).sum())
                    idcg = float(
                        (np.array(ideal) / np.log2(np.arange(len(ideal)) + 2)).sum()
                    )
                    row_metric[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
                per_query.append(row_metric)
            dump(root / f"{name}-per-query.json", per_query)
            metrics[name] = {
                f"ndcg_at_{k}": float(np.mean([q[f"ndcg_at_{k}"] for q in per_query]))
                for k in [5, 10]
            }
            if name == "common":
                replay_docs = [
                    torch.load(root / "replay" / f"{i}.pt", weights_only=True)
                    for i in range(len(rows))
                ]
                replay_scores = streaming_maxsim(
                    bank, replay_docs, query_chunk=32, document_chunk=16
                )
                assert np.array_equal(scores, replay_scores)
                np.save(root / "replay-scores.npy", replay_scores)
                metrics["replay"] = dict(metrics[name])
    dump(
        root / "result.json",
        {
            "metrics": metrics,
            "pages": len(rows),
            "queries": len(queries),
            "all_exact": all(r["exact"] for r in raw),
            "model_load_seconds": load_seconds,
            "one_time_target_check_seconds": check_seconds,
            "trace_seconds": trace_seconds,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "timing_scope": "Warm retained-file cache; image bytes in RAM; decode/preprocess, H2D, boundary hash, state read/integrity, forward/D2H, vector write+fsync. Model load/check and correctness comparison excluded from paired loops. Frozen recipe/image IDs are integration premises.",
            "query_metric": "nDCG@5 and @10, linear graded gains, all standard queries; last qrel row per query/document as official BEIR evaluator; stable corpus-ID ties",
            "smoke_only": bool(args.limit),
        },
    )


if __name__ == "__main__":
    main()
