import numpy as np

from reprforge.token_witness_compiler import (
    compile_token_witnesses,
    matched_random_witnesses,
)


def test_compile_token_witnesses_counts_fit_winners_and_falls_back():
    winners = np.asarray(
        [
            [[0, 1, -1], [2, 2, -1]],
            [[0, 2, -1], [2, 3, -1]],
            [[3, 3, -1], [0, 0, -1]],
        ],
        dtype=np.int16,
    )
    plan = compile_token_witnesses(
        winners,
        fit_queries=(0, 1),
        document_token_counts=(4, 4),
        minimum_win_count=2,
        minimum_tokens=2,
    )
    assert plan[0].tolist() == [0, 1]
    assert plan[1].tolist() == [2, 3]


def test_matched_random_witnesses_preserves_each_size():
    witnesses = (np.asarray([0, 2]), np.asarray([1]))
    random = matched_random_witnesses(
        witnesses, document_token_counts=(4, 3), seed=7
    )
    assert [len(value) for value in random] == [2, 1]
    assert all(len(set(value.tolist())) == len(value) for value in random)


def test_competitive_mask_only_adds_winners_for_active_pairs():
    winners = np.asarray(
        [[[0, 1]], [[2, 3]], [[3, 3]]], dtype=np.int16
    )
    plan = compile_token_witnesses(
        winners,
        fit_queries=(0, 1, 2),
        document_token_counts=(4,),
        minimum_win_count=1,
        minimum_tokens=1,
        competitive_pairs=np.asarray([[True], [False], [False]]),
    )
    assert plan[0].tolist() == [0, 1]
