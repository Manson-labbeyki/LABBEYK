"""
Numerical integrator for the energy-loss equation dE/dx = f(E, x)
"""

from __future__ import annotations
import numpy as np
from .energy_loss import AbstractEnergyLoss


class RK4Integrator:
    """
    Classic 4th-order Runge-Kutta integrator for the ODE:

        dE/dx = loss_rate(E, x)
    """

    def __init__(self, loss_model: AbstractEnergyLoss):
        self.loss = loss_model

    def integrate(self, initial_energy: float,
                  x_span: tuple[float, float],
                  num_steps: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate from x_span[0] to x_span[1].

        Returns
        -------
        x : array of positions [m]
        energy : array of total energy [J]
        """
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        x = np.linspace(x_span[0], x_span[1], num_steps)
        energy = np.zeros(num_steps)
        energy[0] = initial_energy
        dx = x[1] - x[0]

        for i in range(num_steps - 1):
            xi = x[i]
            Ei = energy[i]

            if Ei <= 0.0:
                energy[i:] = 0.0
                break

            k1 = self.loss.loss_rate(Ei, xi)
            k2 = self.loss.loss_rate(Ei + 0.5 * dx * k1, xi + 0.5 * dx)
            k3 = self.loss.loss_rate(Ei + 0.5 * dx * k2, xi + 0.5 * dx)
            k4 = self.loss.loss_rate(Ei + dx * k3, xi + dx)

            energy[i + 1] = Ei + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if energy[i + 1] < 0.0:
                energy[i + 1] = 0.0
                energy[i + 2:] = 0.0
                break

        return x, energy