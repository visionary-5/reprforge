from pathlib import Path

import pytest

from tools.merge_omni_shard_rankings import merge_rankings


def _ranking(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    path.write_text(
        "".join(f"{query}\t{doc}\t{score}\n" for query, doc, score in rows),
        encoding="utf-8",
    )
    return path


def test_merge_is_exact_global_top_k(tmp_path: Path) -> None:
    left = _ranking(
        tmp_path / "left.tsv",
        [("q", "a", 9.0), ("q", "b", 5.0), ("q", "c", 1.0)],
    )
    right = _ranking(
        tmp_path / "right.tsv",
        [("q", "d", 8.0), ("q", "e", 7.0), ("q", "f", 2.0)],
    )
    assert merge_rankings([left, right], top_k=3) == {
        "q": [("a", 9.0), ("d", 8.0), ("e", 7.0)]
    }


def test_merge_rejects_overlapping_corpus_shards(tmp_path: Path) -> None:
    left = _ranking(tmp_path / "left.tsv", [("q", "same", 2.0)])
    right = _ranking(tmp_path / "right.tsv", [("q", "same", 1.0)])
    with pytest.raises(ValueError, match="multiple shards"):
        merge_rankings([left, right], top_k=1)


def test_merge_requires_each_shard_to_cover_global_depth(tmp_path: Path) -> None:
    left = _ranking(tmp_path / "left.tsv", [("q", "a", 2.0)])
    right = _ranking(tmp_path / "right.tsv", [("q", "b", 1.0)])
    with pytest.raises(ValueError, match="at least top_k"):
        merge_rankings([left, right], top_k=2)
