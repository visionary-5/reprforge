"""Protect immutable experiment files when extracting the published evidence."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_artifacts",
    Path(__file__).resolve().parents[1] / "experiments" / "verify_artifacts.py",
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def archive(path, name):
    with tarfile.open(path, "w:gz") as handle:
        member = tarfile.TarInfo(name)
        member.size = 3
        handle.addfile(member, io.BytesIO(b"old"))


def test_extraction_cannot_replace_existing_evidence(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    archive(bundle, "result.json")
    destination = tmp_path / "evidence"
    destination.mkdir()
    result = destination / "result.json"
    result.write_text("original")
    with pytest.raises(FileExistsError):
        VERIFY.extract(bundle, destination)
    assert result.read_text() == "original"


def test_extraction_rejects_path_escape(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    archive(bundle, "../outside.json")
    with pytest.raises(ValueError):
        VERIFY.extract(bundle, tmp_path / "evidence")
    assert not (tmp_path / "outside.json").exists()
