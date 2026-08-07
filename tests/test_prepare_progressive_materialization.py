import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_prepare_progressive_materialization_is_leakage_audited(tmp_path):
    dataset = tmp_path / "dataset"
    assets = dataset / "assets"
    assets.mkdir(parents=True)
    corpus = []
    for index in range(6):
        image = f"p{index}.png"
        Image.new("RGB", (8, 8), (255 - 20 * index,) * 3).save(assets / image)
        corpus.append(
            {
                "docid": f"report-page-{index}",
                "document_id": "report",
                "image": image,
                "text": f"topic {index}",
            }
        )
    queries = [
        {"query_id": f"q{index}", "query": f"topic {index}"} for index in range(6)
    ]
    qrels = [
        {"query_id": f"q{index}", "doc_id": f"report-page-{index}", "relevance": 1}
        for index in range(6)
    ]
    _write_jsonl(dataset / "corpus.jsonl", corpus)
    _write_jsonl(dataset / "queries.jsonl", queries)
    _write_jsonl(dataset / "qrels.jsonl", qrels)
    output = tmp_path / "prepared"
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/prepare_progressive_materialization_v0.py"),
            "--config",
            str(repo / "configs/progressive-visual-materialization-v0.json"),
            "--dataset-root",
            str(dataset),
            "--output-root",
            str(output),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepared_cpu_no_gpu_results" in completed.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset"]["pages"] == 6
    assert manifest["dataset"]["document_grouping_available"] is True
    assert "cheap_locator_disagreement" not in manifest["strategies"]
    assert manifest["traces"]["zipf_1p0"] == 30
    subset = json.loads(
        (
            output
            / "subsets"
            / "risk_cover_plus_history_benefit"
            / "budget-020"
            / "manifest.json"
        ).read_text()
    )
    assert subset["selected_pages"] == 2
    assert subset["information_boundary"]["uses_qrels_for_selection"] is False
    assert subset["direct_physical_build_required"] is True
