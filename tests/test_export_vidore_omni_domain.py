from pathlib import Path

import pytest

from tools.export_vidore_omni_domain import discover_parquet_shards


def test_discover_parquet_shards_keeps_all_sorted_shards(tmp_path: Path):
    for part in ("corpus", "queries", "qrels"):
        root = tmp_path / part
        root.mkdir()
        (root / "test-00001.parquet").touch()
        (root / "test-00000.parquet").touch()

    paths = discover_parquet_shards(tmp_path)

    assert [path.name for path in paths["corpus"]] == [
        "test-00000.parquet",
        "test-00001.parquet",
    ]
    assert all(len(shards) == 2 for shards in paths.values())


def test_discover_parquet_shards_rejects_missing_part(tmp_path: Path):
    for part in ("corpus", "queries"):
        root = tmp_path / part
        root.mkdir()
        (root / "test.parquet").touch()

    with pytest.raises(FileNotFoundError, match="qrels"):
        discover_parquet_shards(tmp_path)
