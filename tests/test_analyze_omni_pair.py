import json

import pytest

from tools.analyze_omni_pair import analyze_pair, load_ranking


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_ranking(path, rows):
    path.write_text("".join(f"{q}\t{d}\t{s}\n" for q, d, s in rows))


def test_analyze_pair_reproduces_metrics_and_finds_escape(tmp_path):
    qrels = tmp_path / "qrels.jsonl"
    full = tmp_path / "full.tsv"
    compressed = tmp_path / "compressed.tsv"
    _write_jsonl(
        qrels,
        [
            {"query_id": "1", "doc_id": "a", "relevance": 1},
            {"query_id": "2", "doc_id": "e", "relevance": 1},
        ],
    )
    _write_ranking(
        full,
        [
            ("1", "a", 3),
            ("1", "b", 2),
            ("1", "c", 1),
            ("2", "e", 3),
            ("2", "f", 2),
            ("2", "g", 1),
        ],
    )
    _write_ranking(
        compressed,
        [
            ("1", "b", 3),
            ("1", "c", 2),
            ("1", "d", 1),
            ("2", "e", 3),
            ("2", "f", 2),
            ("2", "g", 1),
        ],
    )
    report = analyze_pair(
        qrels,
        full,
        compressed,
        depth=3,
        bootstrap_resamples=20,
    )
    assert report["queries"] == 2
    assert report["quality"]["recall_at_1"]["reference_mean"] == 1.0
    assert report["quality"]["recall_at_1"]["candidate_mean"] == 0.5
    assert report["ranking_fidelity"][
        "full_top10_escape_compressed_top100"
    ]["mean"] == pytest.approx(1 / 6)


def test_load_ranking_rejects_increasing_scores(tmp_path):
    path = tmp_path / "ranking.tsv"
    _write_ranking(path, [("q", "a", 1), ("q", "b", 2)])
    with pytest.raises(ValueError, match="scores increase"):
        load_ranking(path, expected_depth=2)
