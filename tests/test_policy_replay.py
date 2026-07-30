from reprforge.policy_replay import (
    IMAGE_ROUTE,
    TEXT_ROUTE,
    Item,
    Query,
    ReplayData,
    RouteCost,
    evaluate_plan,
    exact_budget_oracle,
    fixed_hybrid_plan,
    plan_cost,
    random_plan_under_budget,
    typed_capacity_plan_v1,
    uniform_plan,
)


def _item(item_id: str, content_type: str) -> Item:
    return Item(
        item_id=item_id,
        content_type=content_type,
        route_costs={
            TEXT_ROUTE: RouteCost(index_bytes=1, encode_ms=1.0),
            IMAGE_ROUTE: RouteCost(index_bytes=4, encode_ms=3.0),
        },
    )


def _fixture() -> ReplayData:
    items = (
        _item("text-a", "text"),
        _item("table-b", "table"),
        _item("text-c", "text"),
    )
    queries = (
        Query(query_id="q-text", relevance={"text-a": 1.0}),
        Query(query_id="q-table", relevance={"table-b": 1.0}),
    )
    # Image helps the table but hurts the text item. The fixed hybrid therefore
    # matches the best consistent plan.
    scores = {
        TEXT_ROUTE: {
            "q-text": {"text-a": 0.9, "table-b": 0.3, "text-c": 0.2},
            "q-table": {"text-a": 0.7, "table-b": 0.6, "text-c": 0.5},
        },
        IMAGE_ROUTE: {
            "q-text": {"text-a": 0.1, "table-b": 0.2, "text-c": 0.3},
            "q-table": {"text-a": 0.2, "table-b": 0.95, "text-c": 0.1},
        },
    }
    data = ReplayData(items=items, queries=queries, scores=scores)
    data.validate()
    return data


def test_fixed_hybrid_reproduces_mmdocir_rule() -> None:
    data = _fixture()
    assert fixed_hybrid_plan(data.items) == {
        "text-a": TEXT_ROUTE,
        "table-b": IMAGE_ROUTE,
        "text-c": TEXT_ROUTE,
    }

def test_fixed_hybrid_accepts_a_compressed_visual_route() -> None:
    base = _fixture()
    compressed = "image-pool-3"
    items = tuple(
        Item(
            item_id=item.item_id,
            content_type=item.content_type,
            route_costs={
                **item.route_costs,
                compressed: RouteCost(index_bytes=2, encode_ms=2.0),
            },
        )
        for item in base.items
    )

    plan = fixed_hybrid_plan(items, image_route=compressed)

    assert plan == {
        "text-a": TEXT_ROUTE,
        "table-b": compressed,
        "text-c": TEXT_ROUTE,
    }


def test_typed_capacity_v1_uses_mechanism_derived_routes() -> None:
    routes = {
        TEXT_ROUTE: RouteCost(1, 1.0),
        IMAGE_ROUTE: RouteCost(100, 2.0),
        "image-pool-9": RouteCost(11, 3.0),
        "image-pool-25": RouteCost(4, 4.0),
    }
    items = tuple(
        Item(item_id=name, content_type=kind, route_costs=routes)
        for name, kind in (
            ("table", "table"),
            ("picture", "image"),
            ("paragraph", "text"),
            ("formula", "formula"),
        )
    )

    assert typed_capacity_plan_v1(items) == {
        "table": IMAGE_ROUTE,
        "picture": "image-pool-9",
        "paragraph": "image-pool-25",
        "formula": "image-pool-25",
    }


def test_evaluation_uses_total_relevant_count_for_recall_and_ndcg() -> None:
    data = _fixture()
    result = evaluate_plan(data, fixed_hybrid_plan(data.items), ks=(1, 2))
    assert result["recall_at_1"] == 1.0
    assert result["ndcg_at_1"] == 1.0
    assert result["cost"]["offline_index_bytes"] == 6


def test_random_policy_never_exceeds_budget_and_is_deterministic() -> None:
    data = _fixture()
    first = random_plan_under_budget(data.items, index_budget_bytes=6, seed=17)
    second = random_plan_under_budget(data.items, index_budget_bytes=6, seed=17)
    assert first == second
    assert plan_cost(data.items, first)["offline_index_bytes"] <= 6
    assert sum(route == IMAGE_ROUTE for route in first.values()) == 1


def test_exact_oracle_finds_best_plan_under_budget() -> None:
    data = _fixture()
    plan, result = exact_budget_oracle(
        data,
        index_budget_bytes=6,
        target_metric="recall_at_1",
    )
    assert plan == fixed_hybrid_plan(data.items)
    assert result["recall_at_1"] == 1.0
    assert result["oracle"]["exact"] is True


def test_replay_and_oracle_accept_a_compressed_visual_route() -> None:
    base = _fixture()
    compressed = "image-64"
    items = tuple(
        Item(
            item_id=item.item_id,
            content_type=item.content_type,
            route_costs={
                **item.route_costs,
                compressed: RouteCost(index_bytes=2, encode_ms=2.0),
            },
        )
        for item in base.items
    )
    scores = {
        **base.scores,
        compressed: {
            "q-text": {"text-a": 0.1, "table-b": 0.2, "text-c": 0.3},
            "q-table": {"text-a": 0.2, "table-b": 0.94, "text-c": 0.1},
        },
    }
    data = ReplayData(items=items, queries=base.queries, scores=scores)
    data.validate()

    plan, result = exact_budget_oracle(
        data,
        index_budget_bytes=5,
        target_metric="recall_at_1",
    )

    assert data.routes == ("image", "image-64", "text")
    assert plan["table-b"] == compressed
    assert result["recall_at_1"] == 1.0
    assert result["cost"]["offline_index_bytes"] == 4


def test_exact_oracle_rejects_unbounded_problem() -> None:
    data = _fixture()
    try:
        exact_budget_oracle(data, index_budget_bytes=6, max_items=2)
    except ValueError as exc:
        assert "at most 2 items" in str(exc)
    else:
        raise AssertionError("expected the exact-oracle size guard to fire")


def test_all_text_cost_is_separate_from_online_latency() -> None:
    data = _fixture()
    result = evaluate_plan(data, uniform_plan(data.items, TEXT_ROUTE))
    assert result["cost"] == {
        "offline_index_bytes": 3,
        "offline_encode_ms": 3.0,
        "route_counts": {"text": 3, "image": 0},
    }


def test_query_candidate_pool_limits_within_document_ranking() -> None:
    base = _fixture()
    queries = (
        Query(
            query_id="q-text",
            relevance={"text-a": 1.0},
            candidate_item_ids=("text-a", "table-b"),
        ),
    )
    data = ReplayData(items=base.items, queries=queries, scores=base.scores)
    # text-c has the highest image score for q-text but belongs to another
    # document and therefore must not affect MMDocIR within-document ranking.
    result = evaluate_plan(data, uniform_plan(data.items, IMAGE_ROUTE), ks=(1,))
    assert result["recall_at_1"] == 0.0


def test_within_document_replay_does_not_require_global_score_matrix() -> None:
    base = _fixture()
    query = Query(
        query_id="q-text",
        relevance={"text-a": 1.0},
        candidate_item_ids=("text-a", "table-b"),
    )
    sparse_scores = {
        route: {
            "q-text": {
                item_id: value
                for item_id, value in base.scores[route]["q-text"].items()
                if item_id in query.candidate_item_ids
            }
        }
        for route in (TEXT_ROUTE, IMAGE_ROUTE)
    }
    data = ReplayData(items=base.items, queries=(query,), scores=sparse_scores)
    data.validate()
    assert evaluate_plan(data, uniform_plan(data.items, TEXT_ROUTE), ks=(1,))[
        "recall_at_1"
    ] == 1.0
