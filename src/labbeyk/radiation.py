"""
CMB radiation drag force on a relativistic macroscopic object.
"""

from __future__ import annotations
import numpy as np
from .constants import C, U_CMB


class CMBRadiationDrag:
    """
    Radiation drag force from the Cosmic Microwave Background.
    Uses the standard ultra-relativistic approximation.
    """

    def __init__(self, temperature: float, gamma: float, beta: float,
                 cross_section: float):
        if temperature <= 0:
            raise ValueError("CMB temperature must be positive")
        if gamma < 1.0:
            raise ValueError("gamma must be ≥ 1")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 ≤ β < 1")
        if cross_section <= 0:
            raise ValueError("cross_section must be positive")

        self.T = float(temperature)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.A = float(cross_section)
        self.u_cmb = U_CMB

    def approximate_force(self) -> float:
        """
        Analytic ultra-relativistic radiation drag force:

            F ≈ (4/3) A u_CMB γ² β c
        """
        return (4.0 / 3.0) * self.A * self.u_cmb * (self.gamma ** 2) * self.beta * C

    def total_drag_force(self, method: str = "approx") -> float:
        """Return drag force in Newtons."""
        return self.approximate_force()

    def __repr__(self) -> str:
        return (f"CMBRadiationDrag(T={self.T:.3f} K, γ={self.gamma:.3e}, "
                f"β={self.beta:.6f}, A={self.A:.3e} m²)")