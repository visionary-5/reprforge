import numpy as np

from reprforge.physical_cost import (
    AtomicCostObservation,
    AtomicMaterializationCostModel,
    fit_atomic_cost_model,
)


def test_atomic_cost_model_charges_batches_and_score_events() -> None:
    model = AtomicMaterializationCostModel(
        batch_size=4,
        setup_ms=10.0,
        page_ms=2.0,
        batch_ms=3.0,
        score_event_ms=0.5,
    )
    estimate = model.estimate(pages=5, score_events=8)

    assert estimate.batches == 2
    assert estimate.total_ms == 30.0
    assert model.estimate(pages=0, score_events=0).total_ms == 0.0


def test_fit_atomic_cost_model_recovers_non_negative_signal() -> None:
    expected = AtomicMaterializationCostModel(
        batch_size=4,
        setup_ms=7.0,
        page_ms=2.5,
        batch_ms=4.0,
        score_event_ms=0.2,
    )
    work = [(3, 10), (4, 15), (5, 18), (8, 25), (11, 40), (16, 60)]
    observations = [
        AtomicCostObservation(
            pages=pages,
            score_events=events,
            total_ms=expected.estimate(pages=pages, score_events=events).total_ms,
        )
        for pages, events in work
    ]
    fitted = fit_atomic_cost_model(observations, batch_size=4)

    assert np.allclose(
        [fitted.setup_ms, fitted.page_ms, fitted.batch_ms, fitted.score_event_ms],
        [7.0, 2.5, 4.0, 0.2],
    )

