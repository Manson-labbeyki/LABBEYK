"""
Energy-loss models for a relativistic macroscopic projectile.

All loss rates are expressed as dE/dx in J/m.
A negative value means that the projectile loses energy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .constants import C
from .radiation import CMBRadiationDrag


class AbstractEnergyLoss(ABC):
    """Abstract interface for continuous energy-loss mechanisms."""

    @abstractmethod
    def loss_rate(self, energy: float, position: float) -> float:
        """
        Return dE/dx [J/m].

        Parameters
        ----------
        energy:
            Projectile total energy [J].
        position:
            Propagation distance [m].
        """
        raise NotImplementedError


class IGMEnergyLoss(AbstractEnergyLoss):
    """
    Phenomenological continuous energy-loss model for the IGM.

    The model assumes

        dE/dx = -n * sigma_eff * epsilon * E

    where:
        n           = particle number density [m^-3]
        sigma_eff   = effective interaction cross-section [m^2]
        epsilon     = fractional energy deposition per interaction.

    This is explicitly a phenomenological approximation, not a
    first-principles description of every microscopic interaction.
    """

    def __init__(
        self,
        density: float,
        cross_section: float = 1e-28,
        efficiency: float = 0.05,
    ):
        if density < 0:
            raise ValueError("density must be non-negative")

        if cross_section < 0:
            raise ValueError("cross_section must be non-negative")

        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("efficiency must be between 0 and 1")

        self.density = float(density)
        self.cross_section = float(cross_section)
        self.efficiency = float(efficiency)

    def loss_rate(self, energy: float, position: float) -> float:
        if energy <= 0.0:
            return 0.0

        return (
            -self.density
            * self.cross_section
            * self.efficiency
            * energy
        )


class CMBEnergyLoss(AbstractEnergyLoss):
    """
    Energy loss caused by radiation drag from the CMB.

    The CMB is treated as an isotropic radiation field, and the
    projectile is assigned an effective momentum-transfer cross-section.

    The drag force model is

        F_drag = (4/3) A u_CMB gamma^2 beta c

    and therefore

        dE/dx = -F_drag

    because the drag force acts opposite to the direction of motion.
    """

    def __init__(
        self,
        temperature: float,
        mass: float,
        cross_section: float,
    ):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        if mass <= 0.0:
            raise ValueError("mass must be positive")

        if cross_section <= 0.0:
            raise ValueError("cross_section must be positive")

        self.temperature = float(temperature)
        self.mass = float(mass)
        self.cross_section = float(cross_section)

    def _gamma_beta(self, energy: float) -> tuple[float, float]:
        """
        Convert total relativistic energy into gamma and beta.
        """
        rest_energy = self.mass * C**2

        if energy < rest_energy:
            raise ValueError(
                "Total energy cannot be smaller than the rest energy."
            )

        gamma = energy / rest_energy

        if gamma == 1.0:
            return 1.0, 0.0

        beta = np.sqrt(1.0 - 1.0 / gamma**2)

        return gamma, beta

    def drag_force(self, energy: float) -> float:
        """
        Return the CMB drag-force magnitude [N].
        """
        if energy <= 0.0:
            return 0.0

        gamma, beta = self._gamma_beta(energy)

        drag = CMBRadiationDrag(
            temperature=self.temperature,
            gamma=gamma,
            beta=beta,
            cross_section=self.cross_section,
        )

        return drag.approximate_force()

    def loss_rate(self, energy: float, position: float) -> float:
        """
        Return CMB contribution to dE/dx [J/m].
        """
        return -self.drag_force(energy)


class CombinedEnergyLoss(AbstractEnergyLoss):
    """Combined CMB and IGM energy-loss model."""

    def __init__(
        self,
        igm: IGMEnergyLoss,
        cmb: CMBEnergyLoss,
    ):
        self.igm = igm
        self.cmb = cmb

    def igm_loss_rate(self, energy: float, position: float) -> float:
        """IGM contribution to dE/dx [J/m]."""
        return self.igm.loss_rate(energy, position)

    def cmb_loss_rate(self, energy: float, position: float) -> float:
        """CMB contribution to dE/dx [J/m]."""
        return self.cmb.loss_rate(energy, position)

    def loss_rate(self, energy: float, position: float) -> float:
        """Total dE/dx [J/m]."""
        return (
            self.igm_loss_rate(energy, position)
            + self.cmb_loss_rate(energy, position)
        )