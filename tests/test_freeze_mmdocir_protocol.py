from reprforge.freeze_mmdocir_protocol import freeze_protocol


def test_roles_are_frozen_without_outcome_fields_and_balanced_per_domain() -> None:
    selection = {
        "documents": [
            {
                "document_index": 0,
                "document_name": "old",
                "domain": "a",
                "fixed_pilot": True,
                "layouts": 10,
                "questions": 2,
            },
            {
                "document_index": 1,
                "document_name": "new-1",
                "domain": "a",
                "fixed_pilot": False,
                "layouts": 20,
                "questions": 3,
            },
            {
                "document_index": 2,
                "document_name": "new-2",
                "domain": "a",
                "fixed_pilot": False,
                "layouts": 30,
                "questions": 4,
            },
        ]
    }
    protocol = freeze_protocol(selection, selection_sha256="abc")

    roles = {row["document_index"]: row["role"] for row in protocol["documents"]}
    assert roles[0] == "prior-development"
    assert {roles[1], roles[2]} == {"mechanism-design", "final-evaluation"}
    assert protocol["roles_assigned_without_scores_or_labels"] is True
    assert protocol["scale_contract"]["quality_labels_valid"] is False
