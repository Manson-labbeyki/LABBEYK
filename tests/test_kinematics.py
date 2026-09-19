import numpy as np
import pytest

from src.labbeyk.kinematics import RelativisticKinematics
from src.labbeyk.constants import C


def test_rest_state():
    kin = RelativisticKinematics(mass=20.0, beta=0.0)

    assert kin.gamma == pytest.approx(1.0)
    assert kin.kinetic_energy == pytest.approx(0.0)
    assert kin.momentum == pytest.approx(0.0)


def test_relativistic_gamma():
    kin = RelativisticKinematics(mass=20.0, beta=0.999999)

    assert kin.gamma == pytest.approx(707.106781, rel=1e-6)


def test_energy_momentum_relation():
    kin = RelativisticKinematics(mass=20.0, beta=0.999999)

    lhs = kin.total_energy**2
    rhs = (kin.momentum * C)**2 + (kin.mass * C**2)**2

    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_kinetic_energy_is_total_minus_rest():
    kin = RelativisticKinematics(mass=20.0, beta=0.8)

    rest_energy = kin.mass * C**2

    assert kin.kinetic_energy == pytest.approx(
        kin.total_energy - rest_energy,
        rel=1e-12,
    )


def test_invalid_beta():
    with pytest.raises(ValueError):
        RelativisticKinematics(mass=20.0, beta=1.0)

    with pytest.raises(ValueError):
        RelativisticKinematics(mass=20.0, beta=-0.1)


def test_invalid_mass():
    with pytest.raises(ValueError):
        RelativisticKinematics(mass=0.0, beta=0.5)