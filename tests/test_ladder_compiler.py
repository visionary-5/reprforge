import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.ladder_compiler import (
    _mixed_surface,
    compile_error_bounded_plan,
)


def _cube():
    return ScoreCube(
        query_ids=("q-fit", "q-eval"),
        corpus_ids=("a", "b", "c"),
        scores={
            "cheap": np.asarray([[3.0, 1.9, 1.0], [3.0, 1.9, 1.0]]),
            "mid": np.asarray([[3.0, 2.8, 1.0], [3.0, 2.8, 1.0]]),
            "full": np.asarray([[3.0, 2.9, 1.0], [3.0, 2.9, 1.0]]),
        },
        relevance=({0: 1.0}, {0: 1.0}),
        split_roles=("fit", "eval"),
    )


def test_mixed_surface_uses_one_route_per_item():
    cube = _cube()
    surface = _mixed_surface(cube, ("cheap", "mid", "full"), ((0, 1, 2),) * 2)
    assert np.allclose(surface[0], [3.0, 2.8, 1.0])


def test_error_budget_selects_cheapest_feasible_state():
    cube = _cube()
    plan, bounds = compile_error_bounded_plan(
        cube,
        teacher_route="full",
        candidate_indices=((0, 1, 2),) * 2,
        route_costs={
            "cheap": np.asarray([1.0, 1.0, 1.0]),
            "mid": np.asarray([2.0, 2.0, 2.0]),
            "full": np.asarray([4.0, 4.0, 4.0]),
        },
        error_budget=0.25,
    )
    assert plan[0] == "cheap"
    assert plan[1] == "mid"
    assert bounds[1, cube.routes.index("cheap")] > 0.25
