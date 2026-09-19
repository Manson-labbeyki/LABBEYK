import numpy as np
import pytest

from src.labbeyk.integrator import RK4Integrator


class ExponentialLoss:
    """ODE: dE/dx = -aE"""

    def __init__(self, a):
        self.a = a

    def loss_rate(self, energy, position):
        return -self.a * energy


class ZeroLoss:
    """ODE: dE/dx = 0"""

    def loss_rate(self, energy, position):
        return 0.0


def test_zero_loss_keeps_energy_constant():
    initial_energy = 1e6

    integrator = RK4Integrator(ZeroLoss())

    x, energy = integrator.integrate(
        initial_energy=initial_energy,
        x_span=(0.0, 10.0),
        num_steps=101,
    )

    assert len(x) == 101
    assert np.allclose(energy, initial_energy)


def test_rk4_against_analytic_solution():
    initial_energy = 1e6
    a = 0.01
    distance = 10.0

    integrator = RK4Integrator(ExponentialLoss(a))

    x, energy = integrator.integrate(
        initial_energy=initial_energy,
        x_span=(0.0, distance),
        num_steps=1001,
    )

    expected = initial_energy * np.exp(-a * distance)

    assert energy[-1] == pytest.approx(
        expected,
        rel=1e-8,
    )


def test_invalid_number_of_steps():
    integrator = RK4Integrator(ZeroLoss())

    with pytest.raises(ValueError):
        integrator.integrate(
            initial_energy=1e6,
            x_span=(0.0, 1.0),
            num_steps=1,
        )