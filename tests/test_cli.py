"""Public command help must work without optional GPU packages."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", [
    "scripts/build_source_index.py",
    "scripts/replay_target.py",
    "scripts/evaluate.py",
    "experiments/reconstruction/prepare_inputs.py",
])
def test_help_without_gpu(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_bundled_projection_inputs_match_the_source_manifest():
    manifest = json.loads(
        (ROOT / "experiments/reconstruction/sources.json").read_text()
    )
    bundled = [entry for entry in manifest if "bundled" in entry]
    assert len(bundled) == 4
    for entry in bundled:
        payload = (ROOT / entry["bundled"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
