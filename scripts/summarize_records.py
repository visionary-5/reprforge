"""Summarize recorded measurements without downloading assets or changing files."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


def main():
    root = Path(__file__).resolve().parents[1] / "evidence"
    rows = json.loads((root / "all_rows.json").read_text())
    print("Family-audit records:", len(rows), dict(Counter(r["verdict"] for r in rows)))
    fallback = [r for r in rows if r.get("processor", {}).get("note")]
    print("ACCEPT records using a documented base-processor fallback:", len(fallback))
    for row in fallback:
        print(" ", row["target"], "--", row["processor"]["note"])
    print("Two additional Qwen3 transitions are recorded separately.")
    print("Historical ACCEPT labels do not override processor notes.")
    data = root / "final-pass/quality-full"
    result = json.loads((data / "result.json").read_text())
    print("\nFull HR split quality:")
    for name, metrics in result["metrics"].items():
        print(name, metrics)
    timing = [
        json.loads(line) for line in (data / "timing.jsonl").read_text().splitlines()
    ]
    if len(timing) != 1800 or not all(row["exact"] for row in timing):
        raise ValueError("Unexpected timing-record coverage or exactness flag")
    passes = {}
    for route in ("full", "generic", "fixed"):
        passes[route] = []
        for rep in range(3):
            sample = [r for r in timing if r["route"] == route and r["repeat"] == rep]
            if len(sample) != 200:
                raise ValueError("Each route/pass must contain 200 records")
            passes[route].append(sum(r["total"] for r in sample))
        ms = [s * 1000 / 200 for s in passes[route]]
        print(f"{route}: {mean(ms):.3f} +/- {stdev(ms):.3f} ms/page (three pass means)")
    for route in ("generic", "fixed"):
        denominator = sum(passes[route])
        full = sum(passes["full"])
        charged = denominator + 3 * result["one_time_target_check_seconds"]
        print(
            f"{route}: {full / denominator:.3f}x; "
            f"{full / charged:.3f}x with one check/pass"
        )


if __name__ == "__main__":
    main()
