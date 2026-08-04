import numpy as np

from reprforge.group_policy_compiler import fit_group_policy
from reprforge.heterogeneity_atlas import ScoreCube


def test_group_coordinate_search_retains_capacity_for_evidence_group():
    cube = ScoreCube(
        query_ids=("q0", "q1"),
        corpus_ids=("a", "b"),
        scores={
            "cheap": np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            "image": np.asarray([[2.0, 1.0], [2.0, 1.0]]),
        },
        relevance=({0: 1.0}, {0: 1.0}),
        split_roles=("fit", "fit"),
    )
    result = fit_group_policy(
        cube,
        item_groups=("evidence", "other"),
        candidate_indices=((0, 1), (0, 1)),
        route_costs={"cheap": np.asarray([1.0, 1.0]), "image": np.asarray([3.0, 3.0])},
        fit_mask=(True, True),
        cost_penalty=0.1,
        initial_mappings=({"evidence": "cheap", "other": "cheap"},),
        target_k=1,
    )
    assert result["mapping"] == {"evidence": "image", "other": "cheap"}
