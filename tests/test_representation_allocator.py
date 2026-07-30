import pytest

from reprforge.representation_allocator import RouteOption, allocate_routes


def _options():
    return [
        RouteOption("a", "text", 1, 0.0),
        RouteOption("a", "compressed", 3, 4.0),
        RouteOption("a", "full", 6, 5.0),
        RouteOption("b", "text", 1, 0.0),
        RouteOption("b", "compressed", 3, 3.0),
        RouteOption("b", "full", 6, 10.0),
    ]


def test_allocator_chooses_one_route_per_item_under_budget() -> None:
    allocation = allocate_routes(
        _options(),
        budget_bytes=9,
        cost_quantum_bytes=1,
    )

    assert allocation.plan == {"a": "compressed", "b": "full"}
    assert allocation.total_cost_bytes == 9
    assert allocation.predicted_utility == 14.0
    assert allocation.quantized is False


def test_rounding_up_preserves_the_real_byte_budget() -> None:
    allocation = allocate_routes(
        _options(),
        budget_bytes=8,
        cost_quantum_bytes=4,
    )

    assert allocation.total_cost_bytes <= 8
    assert allocation.cost_quantum_bytes == 4
    assert allocation.quantized is True


def test_allocator_is_deterministic_on_ties() -> None:
    options = [
        RouteOption("x", "a", 1, 1.0),
        RouteOption("x", "b", 1, 1.0),
    ]

    first = allocate_routes(options, budget_bytes=1, cost_quantum_bytes=1)
    second = allocate_routes(options, budget_bytes=1, cost_quantum_bytes=1)

    assert first.plan == second.plan == {"x": "a"}


def test_allocator_rejects_a_budget_below_cheapest_complete_plan() -> None:
    with pytest.raises(ValueError, match="exceeds budget"):
        allocate_routes(_options(), budget_bytes=1)
