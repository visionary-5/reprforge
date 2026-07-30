#!/usr/bin/env python3
"""Select a bounded, domain-stratified MMDocIR expansion without scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PILOT_DOCUMENTS = (2, 18, 82, 90, 100, 104, 139, 220, 230, 277)


def _stable_key(document_index: int, document: Mapping) -> tuple[str, int]:
    identity = (
        f"{document_index}\0{document.get('doc_name', '')}"
        f"\0{document.get('domain', '')}"
    ).encode()
    return hashlib.sha256(identity).hexdigest(), document_index


def _layout_count(document: Mapping) -> int:
    start, end = (int(value) for value in document["layout_indices"])
    return end - start + 1


def select_documents(
    annotations: Sequence[Mapping],
    *,
    target_documents: int,
    max_layouts_per_new_document: int,
    minimum_questions: int,
    fixed_documents: Sequence[int] = PILOT_DOCUMENTS,
) -> dict:
    """Return a deterministic selection independent of retrieval outcomes.

    Fixed pilot documents are retained for continuity. New documents are first
    used to raise every domain toward equal minimum coverage, then allocated to
    domains in proportion to their remaining eligible corpus frequency.
    Within a domain, a SHA-256 ordering over public metadata chooses documents.
    """

    if target_documents < len(set(fixed_documents)):
        raise ValueError("target is smaller than the fixed pilot")
    if max_layouts_per_new_document <= 0 or minimum_questions <= 0:
        raise ValueError("selection bounds must be positive")
    fixed = sorted(set(int(value) for value in fixed_documents))
    if any(index < 0 or index >= len(annotations) for index in fixed):
        raise IndexError("a fixed document index is outside annotations")

    eligible_by_domain: dict[str, list[int]] = defaultdict(list)
    excluded = Counter()
    for index, document in enumerate(annotations):
        if index in fixed:
            continue
        if _layout_count(document) > max_layouts_per_new_document:
            excluded["too_many_layouts"] += 1
            continue
        if len(document["questions"]) < minimum_questions:
            excluded["too_few_questions"] += 1
            continue
        domain = str(document.get("domain") or "")
        eligible_by_domain[domain].append(index)
    for domain, indices in eligible_by_domain.items():
        indices.sort(key=lambda index: _stable_key(index, annotations[index]))

    selected = list(fixed)
    selected_set = set(selected)
    domain_counts = Counter(
        str(annotations[index].get("domain") or "") for index in selected
    )
    all_domains = sorted(
        {str(document.get("domain") or "") for document in annotations}
    )

    # First equalize domain coverage as far as eligibility permits.
    desired_floor = max(1, target_documents // len(all_domains))
    for domain in all_domains:
        while (
            domain_counts[domain] < desired_floor
            and eligible_by_domain.get(domain)
            and len(selected) < target_documents
        ):
            index = eligible_by_domain[domain].pop(0)
            selected.append(index)
            selected_set.add(index)
            domain_counts[domain] += 1

    # Fill remaining capacity by the largest eligible corpus deficit. This is
    # deterministic and does not inspect labels, scores, or model outputs.
    corpus_counts = Counter(
        str(document.get("domain") or "") for document in annotations
    )
    while len(selected) < target_documents:
        candidates = [
            domain
            for domain in all_domains
            if eligible_by_domain.get(domain)
        ]
        if not candidates:
            raise ValueError("not enough eligible documents for target")
        domain = max(
            candidates,
            key=lambda value: (
                corpus_counts[value] / max(domain_counts[value], 1),
                corpus_counts[value],
                value,
            ),
        )
        index = eligible_by_domain[domain].pop(0)
        selected.append(index)
        selected_set.add(index)
        domain_counts[domain] += 1

    rows = []
    for index in sorted(selected):
        document = annotations[index]
        rows.append(
            {
                "document_index": index,
                "fixed_pilot": index in fixed,
                "document_name": str(document.get("doc_name") or ""),
                "domain": str(document.get("domain") or ""),
                "layouts": _layout_count(document),
                "questions": len(document["questions"]),
                "layout_range_inclusive": [
                    int(value) for value in document["layout_indices"]
                ],
            }
        )
    return {
        "selection": "domain-stratified-stable-hash-v1",
        "target_documents": target_documents,
        "fixed_documents": fixed,
        "new_document_max_layouts": max_layouts_per_new_document,
        "minimum_questions": minimum_questions,
        "documents": rows,
        "document_indices": [row["document_index"] for row in rows],
        "new_document_indices": [
            row["document_index"] for row in rows if not row["fixed_pilot"]
        ],
        "domain_counts": dict(
            sorted(Counter(row["domain"] for row in rows).items())
        ),
        "layout_count": sum(row["layouts"] for row in rows),
        "question_count": sum(row["questions"] for row in rows),
        "excluded_new_candidates": dict(sorted(excluded.items())),
        "uses_retrieval_labels_or_scores": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--target-documents", type=int, default=30)
    parser.add_argument("--max-layouts-per-new-document", type=int, default=400)
    parser.add_argument("--minimum-questions", type=int, default=2)
    parser.add_argument(
        "--fixed-document",
        type=int,
        action="append",
        dest="fixed_documents",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = [
        json.loads(line)
        for line in args.annotations_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = select_documents(
        annotations,
        target_documents=args.target_documents,
        max_layouts_per_new_document=args.max_layouts_per_new_document,
        minimum_questions=args.minimum_questions,
        fixed_documents=(
            args.fixed_documents
            if args.fixed_documents is not None
            else PILOT_DOCUMENTS
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
