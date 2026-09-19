import pytest

from src.labbeyk.energy_loss import (
    IGMEnergyLoss,
    CMBEnergyLoss,
    CombinedEnergyLoss,
)
from src.labbeyk.constants import C


def test_igm_loss_is_negative():
    model = IGMEnergyLoss(
        density=1.0,
        cross_section=1e-28,
        efficiency=0.05,
    )

    energy = 1e21
    loss = model.loss_rate(energy, 0.0)

    assert loss < 0.0


def test_igm_loss_formula():
    model = IGMEnergyLoss(
        density=1.0,
        cross_section=1e-28,
        efficiency=0.05,
    )

    energy = 1e21

    expected = -1.0 * 1e-28 * 0.05 * energy

    assert model.loss_rate(energy, 0.0) == pytest.approx(expected)


def test_cmb_loss_is_negative():
    model = CMBEnergyLoss(
        temperature=2.725,
        mass=20.0,
        cross_section=0.01,
    )

    energy = 707.106781 * 20.0 * C**2

    assert model.loss_rate(energy, 0.0) < 0.0


def test_combined_loss_equals_sum():
    igm = IGMEnergyLoss(
        density=1.0,
        cross_section=1e-28,
        efficiency=0.05,
    )

    cmb = CMBEnergyLoss(
        temperature=2.725,
        mass=20.0,
        cross_section=0.01,
    )

    combined = CombinedEnergyLoss(igm=igm, cmb=cmb)

    energy = 1e21

    expected = (
        igm.loss_rate(energy, 0.0)
        + cmb.loss_rate(energy, 0.0)
    )

    assert combined.loss_rate(energy, 0.0) == pytest.approx(expected)