import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.marginal_policy_compiler import (
    crossfit_marginal_policy_compiler,
    single_switch_gains,
)


def _cube() -> ScoreCube:
    return ScoreCube(
        query_ids=("q0", "q1", "q2", "q3"),
        corpus_ids=("a", "b", "c"),
        scores={
            "image": np.asarray(
                [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 3.0, 2.0]]
            ),
            "image-pool-9": np.asarray(
                [[2.0, 3.0, 1.0], [2.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
            ),
            "text": np.asarray(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [3.0, 1.0, 2.0]]
            ),
        },
        relevance=({0: 1.0}, {0: 1.0}, {1: 1.0}, {1: 1.0}),
        split_roles=("fit", "fit", "eval", "eval"),
    )


def test_single_switch_gain_identifies_helpful_full_items():
    gains = single_switch_gains(_cube(), base_route="image-pool-9", target_k=1)
    assert np.all(gains["image"][:2, 0] > 0.0)
    assert np.all(gains["image"][2:, 1] > 0.0)


def test_crossfit_marginal_compiler_beats_base_on_repeated_workload():
    cube = _cube()
    costs = {
        "image": np.asarray([2.0, 2.0, 2.0]),
        "image-pool-9": np.asarray([1.0, 1.0, 1.0]),
        "text": np.asarray([1.5, 1.5, 1.5]),
    }
    result = crossfit_marginal_policy_compiler(
        cube,
        base_route="image-pool-9",
        route_costs=costs,
        fold_ids=(0, 1, 0, 1),
        budget_fractions=(1.0,),
        target_k=1,
    )
    assert result["budget_curve"]["1.0"]["crossfit_ndcg_at_5"] == 1.0
