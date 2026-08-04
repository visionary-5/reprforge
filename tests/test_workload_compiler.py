import numpy as np

from reprforge.workload_compiler import _budgeted_documents, _fidelity_for_mask


def test_budgeted_documents_respects_cost_and_utility_density():
    utility = np.asarray([4.0, 3.0, 1.0])
    costs = np.asarray([4.0, 1.0, 1.0])
    selected = _budgeted_documents(utility, costs, budget=2.0)
    assert selected == {1, 2}
    assert costs[list(selected)].sum() <= 2.0


def test_fidelity_for_mask_is_bounded():
    x = np.asarray([2.0, 1.0, 0.0, -1.0])
    expensive = np.asarray([0.0, 2.0, 1.0, -1.0])
    observed = np.asarray([True, True, False, True])
    teacher_top = np.asarray([1, 0])
    fidelity = _fidelity_for_mask(x, expensive, observed, teacher_top, 2)
    assert 0.0 <= fidelity <= 1.0
