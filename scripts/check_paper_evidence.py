"""Recompute lifecycle aggregates from recorded summaries on CPU, without mutation."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence/lifecycle"
    expected = json.loads((evidence / "analysis.json").read_text())["lifecycle"]
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for source in evidence.glob("*.json"):
            if source.name != "analysis.json":
                shutil.copyfile(source, work / source.name)
        subprocess.run(
            [
                sys.executable,
                str(root / "experiments/paper/e2e/analyze_e2e.py"),
                str(work),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = json.loads((work / "analysis.json").read_text())["lifecycle"]
        if actual != expected:
            raise SystemExit("Lifecycle aggregates differ from recorded analysis.")
    rows = json.loads((root / "evidence/all_rows.json").read_text())
    print(
        f"Lifecycle aggregates match the original records; {expected['pages']} pages."
    )
    qwen = [
        json.loads((root / "evidence/out-qwen3-2b/summary.json").read_text())["rudore"],
        json.loads((root / "evidence/out-qwen3-4b/summary.json").read_text())["result"],
    ]
    if len(rows) != 74 or any(
        not x["model_level_ok"] or x["generic_bitwise"] != x["pages"] for x in qwen
    ):
        raise SystemExit(
            "Released-pair evidence coverage differs from the documented protocol."
        )
    print("Included released-pair records: 74 sweep pairs + 2 Qwen3 transitions = 76.")
    print("This check verifies aggregate calculations, not the native-processor label.")
    print("See docs/protocol-notes.md for recorded processor fallbacks.")
    print("No GPU experiment was run; full vectors are not included.")


if __name__ == "__main__":
    main()
