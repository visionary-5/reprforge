"""Cross-collection transfer metrics for qrel-free compression gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean


def summarize_gate_transfer(records: Sequence[Mapping]) -> dict:
    """Summarize certificate decisions against later relevance safety.

    Every record describes one dataset/configuration pair. Full is an implicit
    safe state with resident fraction 1 and zero full-relative regret.
    """

    if not records:
        raise ValueError("gate-transfer records must be non-empty")
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        dataset = str(record["dataset"])
        candidate = str(record["candidate"])
        key = (dataset, candidate)
        if key in seen:
            raise ValueError(f"duplicate gate-transfer record {key}")
        seen.add(key)
        fraction = float(record["resident_fraction"])
        regret = float(record["mean_ndcg10_regret"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("resident fraction must lie in (0, 1]")
        normalized.append(
            {
                "dataset": dataset,
                "candidate": candidate,
                "certificate_passes": bool(record["certificate_passes"]),
                "safety_passes": bool(record["safety_passes"]),
                "resident_fraction": fraction,
                "mean_ndcg10_regret": regret,
            }
        )

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for record in normalized:
        by_dataset[record["dataset"]].append(record)

    selections: dict[str, dict] = {}
    oracle_selections: dict[str, dict] = {}
    for dataset, candidates in sorted(by_dataset.items()):
        accepted = [value for value in candidates if value["certificate_passes"]]
        safe = [value for value in candidates if value["safety_passes"]]
        selected = min(accepted, key=lambda value: value["resident_fraction"]) if accepted else None
        oracle = min(safe, key=lambda value: value["resident_fraction"]) if safe else None
        selections[dataset] = {
            "candidate": "full" if selected is None else selected["candidate"],
            "resident_fraction": 1.0 if selected is None else selected["resident_fraction"],
            "mean_ndcg10_regret": 0.0 if selected is None else selected["mean_ndcg10_regret"],
            "safety_passes": True if selected is None else selected["safety_passes"],
        }
        oracle_selections[dataset] = {
            "candidate": "full" if oracle is None else oracle["candidate"],
            "resident_fraction": 1.0 if oracle is None else oracle["resident_fraction"],
        }

    candidate_names = {record["candidate"] for record in normalized}
    fixed_states = [{"candidate": "full", "macro_resident_fraction": 1.0}]
    for candidate in sorted(candidate_names):
        values = [
            record for record in normalized if record["candidate"] == candidate
        ]
        if len(values) == len(by_dataset) and all(
            value["safety_passes"] for value in values
        ):
            fixed_states.append(
                {
                    "candidate": candidate,
                    "macro_resident_fraction": mean(
                        value["resident_fraction"] for value in values
                    ),
                }
            )
    best_fixed = min(
        fixed_states, key=lambda value: value["macro_resident_fraction"]
    )
    false_safe = sum(
        record["certificate_passes"] and not record["safety_passes"]
        for record in normalized
    )
    false_reject = sum(
        not record["certificate_passes"] and record["safety_passes"]
        for record in normalized
    )
    matches = len(normalized) - false_safe - false_reject
    selected_values = list(selections.values())
    oracle_values = list(oracle_selections.values())
    return {
        "datasets": len(by_dataset),
        "configurations": len(normalized),
        "decision_matches": matches,
        "decision_accuracy": matches / len(normalized),
        "false_safe": false_safe,
        "false_safe_rate": false_safe / len(normalized),
        "false_reject": false_reject,
        "false_reject_rate": false_reject / len(normalized),
        "all_selected_states_safe": all(
            value["safety_passes"] for value in selected_values
        ),
        "selections": selections,
        "oracle_selections": oracle_selections,
        "macro_selected_resident_fraction": mean(
            value["resident_fraction"] for value in selected_values
        ),
        "macro_oracle_safe_resident_fraction": mean(
            value["resident_fraction"] for value in oracle_values
        ),
        "best_fixed_safe_state": best_fixed,
        "macro_selected_mean_ndcg10_regret": mean(
            value["mean_ndcg10_regret"] for value in selected_values
        ),
    }
