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


def test_drag_has_expected_gamma_squared_scaling():
    """
    In the ultra-relativistic approximation:

        F ∝ gamma² * beta
    """

    gamma1 = 100.0
    gamma2 = 200.0

    beta1 = (1.0 - 1.0 / gamma1**2) ** 0.5
    beta2 = (1.0 - 1.0 / gamma2**2) ** 0.5

    drag1 = CMBRadiationDrag(
        temperature=2.725,
        gamma=gamma1,
        beta=beta1,
        cross_section=0.01,
    ).approximate_force()

    drag2 = CMBRadiationDrag(
        temperature=2.725,
        gamma=gamma2,
        beta=beta2,
        cross_section=0.01,
    ).approximate_force()

    expected_ratio = (
        gamma2**2 * beta2
    ) / (
        gamma1**2 * beta1
    )

    assert drag2 / drag1 == pytest.approx(
        expected_ratio,
        rel=1e-12,
    )

