import numpy as np

from reprforge.closure_materialization import (
    ClosureCompilerConfig,
    compile_closure_plan,
    plan_query_closure,
)


def test_compiler_selects_reused_pages_without_future_rows() -> None:
    candidates = np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [4, 5],  # held-out row must not affect compilation
        ]
    )
    plan = compile_closure_plan(
        candidates,
        [0, 1, 2],
        config=ClosureCompilerConfig(
            persistent_page_budget=2,
            expected_future_queries=6,
        ),
    )
    assert plan["persistent_pages"][0] == 0
    assert 4 not in plan["persistent_pages"]


def test_query_plan_completes_missing_comparison_support() -> None:
    plan = plan_query_closure([3, 2, 1], [2, 9])
    assert plan["persistent_pages"] == [2]
    assert plan["transient_pages"] == [3, 1]
    assert plan["candidate_closure_complete"] is True
