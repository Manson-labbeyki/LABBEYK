"""
High-level relativistic projectile simulation.
"""

from __future__ import annotations

import numpy as np

from .constants import C, SimulationParameters
from .energy_loss import (
    CombinedEnergyLoss,
    CMBEnergyLoss,
    IGMEnergyLoss,
)
from .integrator import RK4Integrator
from .kinematics import RelativisticKinematics
from .radiation import CMBRadiationDrag


class RelativisticProjectile:
    """
    Complete simulation of a relativistic macroscopic projectile.
    """

    def __init__(
        self,
        params: SimulationParameters | None = None,
    ):
        self.params = (
            params if params is not None else SimulationParameters()
        )
        self.kin = RelativisticKinematics(
            self.params.mass,
            self.params.beta0,
        )

    @property
    def initial_energy(self) -> float:
        """Initial total energy [J]."""
        return self.kin.total_energy

    @property
    def initial_gamma(self) -> float:
        """Initial Lorentz factor."""
        return self.kin.gamma

    @property
    def initial_kinetic_energy(self) -> float:
        """Initial kinetic energy [J]."""
        return self.kin.kinetic_energy

    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius corresponding to the mass [m]."""
        return self.kin.schwarzschild_radius()

    def _igm_loss(self) -> IGMEnergyLoss:
        """Construct the configured IGM loss model."""
        return IGMEnergyLoss(
            density=self.params.n_igm,
            cross_section=self.params.igm_cross_section,
            efficiency=self.params.igm_efficiency,
        )

    def _cmb_loss(self) -> CMBEnergyLoss:
        """Construct the configured CMB loss model."""
        return CMBEnergyLoss(
            temperature=self.params.t_cmb,
            mass=self.params.mass,
            cross_section=self.params.cross_section,
        )

    def _combined_loss(self) -> CombinedEnergyLoss:
        """Construct the combined CMB + IGM model."""
        return CombinedEnergyLoss(
            igm=self._igm_loss(),
            cmb=self._cmb_loss(),
        )

    def _integrate(self, loss_model, num_steps: int):
        """Integrate one energy-loss model."""
        integrator = RK4Integrator(loss_model)

        return integrator.integrate(
            initial_energy=self.initial_energy,
            x_span=(0.0, self.params.distance_m),
            num_steps=num_steps,
        )

    def simulate(
        self,
        num_steps: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the combined CMB + IGM simulation.
        """
        return self._integrate(
            self._combined_loss(),
            num_steps,
        )

    def simulate_cmb_only(
        self,
        num_steps: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a CMB-only counterfactual simulation.
        """
        return self._integrate(
            self._cmb_loss(),
            num_steps,
        )

    def simulate_igm_only(
        self,
        num_steps: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run an IGM-only counterfactual simulation.
        """
        return self._integrate(
            self._igm_loss(),
            num_steps,
        )

    def initial_cmb_drag(self) -> float:
        """Initial CMB drag force [N]."""
        return CMBRadiationDrag(
            temperature=self.params.t_cmb,
            gamma=self.kin.gamma,
            beta=self.kin.beta,
            cross_section=self.params.cross_section,
        ).total_drag_force(method="approx")

    def final_state(self, energy: float) -> dict[str, float]:
        """
        Calculate relativistic quantities from a final total energy.
        """
        rest_energy = self.params.mass * C**2

        gamma = energy / rest_energy

        if gamma < 1.0:
            gamma = 1.0

        beta = np.sqrt(
            max(0.0, 1.0 - 1.0 / gamma**2)
        )

        kinetic_energy = energy - rest_energy

        return {
            "energy": energy,
            "kinetic_energy": kinetic_energy,
            "gamma": gamma,
            "beta": beta,
        }