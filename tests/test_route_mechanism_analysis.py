from reprforge.policy_replay import Item, Query, ReplayData, RouteCost
from reprforge.route_mechanism_analysis import (
    aggregate_item_routes,
    diagnostic_correlations,
    intervention_rows,
    subset_by_documents,
)


def _data() -> ReplayData:
    costs = {
        "base": RouteCost(10, 1.0),
        "rich": RouteCost(20, 2.0),
    }
    data = ReplayData(
        items=(
            Item("relevant", "text", costs),
            Item("distractor", "image", costs),
            Item("other", "text", costs),
        ),
        queries=(
            Query(
                "q",
                {"relevant": 1.0},
                candidate_item_ids=("relevant", "distractor", "other"),
            ),
        ),
        scores={
            "base": {
                "q": {
                    "relevant": 0.8,
                    "distractor": 0.7,
                    "other": 0.6,
                }
            },
            "rich": {
                "q": {
                    "relevant": 0.5,
                    "distractor": 0.9,
                    "other": 0.4,
                }
            },
        },
    )
    data.validate()
    return data


def test_interventions_separate_evidence_and_distractor_channels() -> None:
    rows = intervention_rows(_data(), base_route="base", metric="ndcg", k=1)
    by_item = {row["item_id"]: row for row in rows}

    assert by_item["relevant"]["channel"] == "evidence_loss"
    assert by_item["distractor"]["channel"] == "distractor_inflation"
    assert by_item["distractor"]["entered_top_k"]
    assert by_item["other"]["channel"] == "no_effect"

    aggregates = aggregate_item_routes(rows)
    distractor = next(
        row for row in aggregates if row["item_id"] == "distractor"
    )
    assert distractor["distractor_harm_sum"] == 1.0
    assert distractor["impact_probability"] == 1.0


def test_diagnostic_correlations_ignore_nonmechanistic_fields() -> None:
    aggregates = [
        {
            "item_id": f"i{index}",
            "route": "pool",
            "impact_probability": value,
            "signed_metric_delta_sum": -value,
            "evidence_delta_sum": 0.0,
            "distractor_harm_sum": value,
        }
        for index, value in enumerate((0.0, 0.5, 1.0))
    ]
    items = [
        {
            "item_id": f"i{index}",
            "route_features": {
                "pool": {
                    "cosine_cover_loss_max": value,
                    "vector_count": 10,
                },
            },
            "construction_features": {"edge_energy": value},
        }
        for index, value in enumerate((0.0, 0.5, 1.0))
    ]
    rows = diagnostic_correlations(aggregates, items)

    assert all("vector_count" not in row["feature"] for row in rows)
    impact = next(
        row
        for row in rows
        if row["outcome"] == "impact_probability"
        and row["feature"] == "route:cosine_cover_loss_max"
    )
    assert impact["spearman"] == 1.0
    construction = next(
        row
        for row in rows
        if row["outcome"] == "distractor_harm_sum"
        and row["feature"] == "construction:edge_energy"
    )
    assert construction["spearman"] == 1.0


def test_subset_by_documents_preserves_official_candidate_pools() -> None:
    data = _data()
    subset, rows = subset_by_documents(
        data,
        query_rows=[{"query_id": "q", "document_index": 7}],
        item_rows=[
            {"item_id": "relevant"},
            {"item_id": "distractor"},
            {"item_id": "other"},
            {"item_id": "outside"},
        ],
        document_indices={7},
    )

    assert len(subset.queries) == 1
    assert {item.item_id for item in subset.items} == {
        "relevant",
        "distractor",
        "other",
    }
    assert {row["item_id"] for row in rows} == {
        "relevant",
        "distractor",
        "other",
    }
