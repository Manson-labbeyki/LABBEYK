import pytest

from src.labbeyk.radiation import CMBRadiationDrag


def test_zero_drag_at_rest():
    drag = CMBRadiationDrag(
        temperature=2.725,
        gamma=1.0,
        beta=0.0,
        cross_section=0.01,
    )

    assert drag.approximate_force() == pytest.approx(0.0)


def test_drag_is_positive_for_motion():
    drag = CMBRadiationDrag(
        temperature=2.725,
        gamma=10.0,
        beta=(1.0 - 1.0 / 100.0) ** 0.5,
        cross_section=0.01,
    )

    assert drag.approximate_force() > 0.0


def test_drag_increases_with_gamma():
    drag_low = CMBRadiationDrag(
        temperature=2.725,
        gamma=10.0,
        beta=(1.0 - 1.0 / 100.0) ** 0.5,
        cross_section=0.01,
    )

    drag_high = CMBRadiationDrag(
        temperature=2.725,
        gamma=20.0,
        beta=(1.0 - 1.0 / 400.0) ** 0.5,
        cross_section=0.01,
    )

    assert drag_high.approximate_force() > drag_low.approximate_force()


def test_drag_scales_with_cross_section():
    drag_small = CMBRadiationDrag(
        temperature=2.725,
        gamma=10.0,
        beta=(1.0 - 1.0 / 100.0) ** 0.5,
        cross_section=0.01,
    )

    drag_large = CMBRadiationDrag(
        temperature=2.725,
        gamma=10.0,
        beta=(1.0 - 1.0 / 100.0) ** 0.5,
        cross_section=0.02,
    )

    assert drag_large.approximate_force() == pytest.approx(
        2.0 * drag_small.approximate_force(),
        rel=1e-12,
    )