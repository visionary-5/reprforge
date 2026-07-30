#!/usr/bin/env python3
"""Freeze design/evaluation roles before inspecting expansion scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping


def _stable_key(document: Mapping) -> tuple[str, int]:
    identity = (
        f"{int(document['document_index'])}\0"
        f"{document.get('document_name', '')}\0"
        f"{document.get('domain', '')}\0reprforge-evaluation-v1"
    ).encode()
    return hashlib.sha256(identity).hexdigest(), int(document["document_index"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol(selection: Mapping, *, selection_sha256: str) -> dict:
    """Assign roles using public metadata only.

    Existing pilot documents are development data because their outcomes have
    already influenced the design. New documents are independently hash-sorted
    within each domain and alternated between design and final evaluation.
    """

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    roles: dict[int, str] = {}
    for document in selection["documents"]:
        index = int(document["document_index"])
        if bool(document["fixed_pilot"]):
            roles[index] = "prior-development"
        else:
            grouped[str(document["domain"])].append(document)
    for domain, documents in grouped.items():
        documents.sort(key=_stable_key)
        for position, document in enumerate(documents):
            roles[int(document["document_index"])] = (
                "mechanism-design" if position % 2 == 0 else "final-evaluation"
            )

    documents = [
        {**dict(document), "role": roles[int(document["document_index"])]}
        for document in selection["documents"]
    ]
    role_counts = Counter(document["role"] for document in documents)
    role_layouts = Counter()
    role_questions = Counter()
    role_domains: dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        role = str(document["role"])
        role_layouts[role] += int(document["layouts"])
        role_questions[role] += int(document["questions"])
        role_domains[role][str(document["domain"])] += 1
    return {
        "protocol": "reprforge-mmdocir-expansion-v1",
        "selection_sha256": selection_sha256,
        "roles_assigned_without_scores_or_labels": True,
        "documents": documents,
        "role_counts": dict(sorted(role_counts.items())),
        "role_layout_counts": dict(sorted(role_layouts.items())),
        "role_question_counts": dict(sorted(role_questions.items())),
        "role_domain_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_domains.items())
        },
        "quality_contract": {
            "candidate_pool": "official within-document MMDocIR candidates",
            "prior-development": (
                "may be used for implementation and prior design; never reported "
                "as untouched evidence"
            ),
            "mechanism-design": (
                "may be inspected for explanatory analysis and V1 design"
            ),
            "final-evaluation": (
                "must remain sealed until V1 features, allocator, and budgets "
                "are frozen"
            ),
            "primary_metrics": ["ndcg_at_10", "recall_at_5", "recall_at_10"],
            "required_baselines": [
                "uniform-image-pool-4",
                "uniform-image-pool-9",
                "uniform-image-pool-25",
                "uniform-image",
                "fixed-hybrid",
                "ReprForge-V0",
            ],
        },
        "scale_contract": {
            "method": "physical vector replication with distinct identifiers",
            "quality_labels_valid": False,
            "allowed_claim": "latency, throughput, memory, and scaling only",
            "forbidden_claim": (
                "replicated unjudged candidates cannot support retrieval quality"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    protocol = freeze_protocol(
        selection,
        selection_sha256=_sha256(args.selection),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
