import hashlib
import json

import pytest

from tools.materialize_vidore_labels import _load_certificates


def _certificate(*, qrels_loaded=False, runtime_sha256="runtime-hash"):
    return {
        "stage": "pre-qrel-ranking-certification",
        "dataset": "validation-dataset",
        "candidate": "pool4",
        "qrels_loaded": qrels_loaded,
        "artifacts": {"reference_runtime_sha256": runtime_sha256},
    }


def test_label_stage_accepts_and_hashes_matching_pre_qrel_certificate(tmp_path):
    path = tmp_path / "pool4-certificate.json"
    payload = json.dumps(_certificate(), sort_keys=True)
    path.write_text(payload)
    observed = _load_certificates(
        [path],
        dataset="validation-dataset",
        runtime_sha256="runtime-hash",
    )
    assert observed == {
        "pool4": hashlib.sha256(payload.encode()).hexdigest()
    }


@pytest.mark.parametrize(
    ("certificate", "error"),
    [
        (_certificate(qrels_loaded=True), "qrels_loaded=false"),
        (_certificate(runtime_sha256="other"), "different full runtime"),
    ],
)
def test_label_stage_rejects_invalid_information_boundary(
    tmp_path, certificate, error
):
    path = tmp_path / "invalid-certificate.json"
    path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError, match=error):
        _load_certificates(
            [path],
            dataset="validation-dataset",
            runtime_sha256="runtime-hash",
        )
