import numpy as np

from reprforge.heterogeneity_atlas import ScoreCube
from reprforge.type_policy_compiler import crossfit_type_policy_compiler


def test_crossfit_type_policy_finds_the_useful_route_under_budget():
    cube = ScoreCube(
        query_ids=("q0", "q1", "q2", "q3"),
        corpus_ids=("a", "b"),
        scores={
            "cheap": np.asarray([[0.0, 1.0]] * 4),
            "full": np.asarray([[2.0, 1.0]] * 4),
        },
        relevance=({0: 1.0},) * 4,
        split_roles=("fit",) * 4,
    )
    result = crossfit_type_policy_compiler(
        cube,
        item_types=("evidence", "other"),
        candidate_indices=((0, 1),) * 4,
        route_costs={"cheap": np.asarray([1.0, 1.0]), "full": np.asarray([3.0, 3.0])},
        fold_ids=(0, 1, 0, 1),
        budget_fractions=(2.0 / 3.0,),
        teacher_route="full",
        target_k=1,
    )
    policy = result["budget_curve"][str(2.0 / 3.0)]
    assert policy["crossfit_ndcg_at_5"] == 1.0
    assert all(
        fold["mapping"] == {"evidence": "full", "other": "cheap"}
        for fold in policy["selected_by_fold"]
    )
