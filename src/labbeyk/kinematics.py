"""
Relativistic kinematics of the projectile.
"""

from __future__ import annotations
import numpy as np
from .constants import C, G_GRAV


class RelativisticKinematics:
    """
    Special-relativistic kinematics for a massive particle.
    """

    def __init__(self, mass: float, beta: float):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 ≤ β < 1")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        self._mass = float(mass)
        self._beta = float(beta)
        self._gamma = 1.0 / np.sqrt(1.0 - beta * beta)

    # ---------- properties ----------
    @property
    def mass(self) -> float:
        return self._mass

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def total_energy(self) -> float:
        """Total energy E = γ m c² [Joule]."""
        return self._gamma * self._mass * C * C

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy (γ − 1) m c² [Joule]."""
        return (self._gamma - 1.0) * self._mass * C * C

    @property
    def momentum(self) -> float:
        """Relativistic momentum γ m β c [kg·m/s]."""
        return self._gamma * self._mass * self._beta * C

    # ---------- convenience methods ----------
    def energy_in_MeV(self) -> float:
        """Total energy in MeV."""
        return self.total_energy / 1.602176634e-13

    def kinetic_energy_TNT_tons(self) -> float:
        """Kinetic energy expressed in tons of TNT equivalent."""
        return self.kinetic_energy / 4.184e9

    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius if the mass were a black hole [m]."""
        return 2.0 * G_GRAV * self._mass / (C * C)

    def __repr__(self) -> str:
        return (f"RelativisticKinematics(mass={self._mass:.3g} kg, "
                f"β={self._beta:.6f}, γ={self._gamma:.4e})")