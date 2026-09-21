import pytest

from src.labbeyk.constants import SimulationParameters
from src.labbeyk.projectile import RelativisticProjectile


def test_projectile_initial_state():
    params = SimulationParameters()
    projectile = RelativisticProjectile(params)

    assert projectile.initial_energy > 0.0
    assert projectile.initial_gamma > 1.0
    assert projectile.initial_kinetic_energy > 0.0


def test_cmb_drag_is_positive():
    projectile = RelativisticProjectile()

    assert projectile.initial_cmb_drag() > 0.0


def test_three_simulations_have_valid_energy():
    projectile = RelativisticProjectile()

    _, e_igm = projectile.simulate_igm_only(num_steps=1000)
    _, e_cmb = projectile.simulate_cmb_only(num_steps=1000)
    _, e_combined = projectile.simulate(num_steps=1000)

    initial = projectile.initial_energy

    assert e_igm[-1] > 0.0
    assert e_cmb[-1] > 0.0
    assert e_combined[-1] > 0.0

    assert e_igm[-1] <= initial
    assert e_cmb[-1] <= initial
    assert e_combined[-1] <= initial


def test_combined_loss_cannot_be_smaller_than_each_loss_channel():
    projectile = RelativisticProjectile()

    _, e_igm = projectile.simulate_igm_only(num_steps=1000)
    _, e_cmb = projectile.simulate_cmb_only(num_steps=1000)
    _, e_combined = projectile.simulate(num_steps=1000)

    assert e_combined[-1] <= e_igm[-1]
    assert e_combined[-1] <= e_cmb[-1]


def test_energy_monotonically_decreases():
    projectile = RelativisticProjectile()

    _, energy = projectile.simulate(num_steps=1000)

    differences = energy[1:] - energy[:-1]

    assert (differences <= 0.0).all()

