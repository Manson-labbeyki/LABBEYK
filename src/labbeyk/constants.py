"""
Physical constants and default simulation parameters.
All quantities use SI units.
"""

from dataclasses import dataclass

import numpy as np


# ============================================================
# Fundamental physical constants — SI
# ============================================================

C = 299_792_458.0                  # speed of light [m/s]
H = 6.626_070_15e-34               # Planck constant [J s]
K_B = 1.380_649e-23                # Boltzmann constant [J/K]
SIGMA_SB = 5.670_374_419e-8        # Stefan-Boltzmann constant [W/m²/K⁴]
G_GRAV = 6.674_30e-11               # gravitational constant [m³/kg/s²]
PARSEC = 3.085_677_581_491_367e16  # parsec [m]


# ============================================================
# Default simulation parameters
# ============================================================

M_OBJECT = 20.0                     # projectile mass [kg]
DISTANCE_MPC = 20.0                 # propagation distance [Mpc]
BETA_0 = 0.999999                   # initial velocity / c
T_CMB = 2.725                       # CMB temperature [K]

# ~1 particle / m³ = 10^-6 particles / cm³
N_IGM = 1.0                          # IGM number density [m⁻³]
IGM_CROSS_SECTION = 1e-28
IGM_EFFICIENCY = 0.05


# ============================================================
# Derived quantities
# ============================================================

DISTANCE_METERS = DISTANCE_MPC * 1e6 * PARSEC

# Blackbody radiation energy density:
#
#     u = a T^4
#       = (4 sigma / c) T^4
#
U_CMB = 4.0 * SIGMA_SB * T_CMB**4 / C


@dataclass(frozen=True)
class SimulationParameters:
    """Immutable container for the simulation parameters."""

    mass: float = M_OBJECT
    distance_m: float = DISTANCE_METERS
    beta0: float = BETA_0
    t_cmb: float = T_CMB
    n_igm: float = N_IGM
    igm_cross_section: float = IGM_CROSS_SECTION
    igm_efficiency: float = IGM_EFFICIENCY

    # Effective geometric cross-section of the projectile [m²].
    #
    # This is a modelling assumption and will be documented
    # explicitly in the scientific report.
    cross_section: float = 0.01

    @property
    def gamma0(self) -> float:
        """Initial Lorentz factor."""
        return 1.0 / np.sqrt(1.0 - self.beta0**2)

    @property
    def u_cmb(self) -> float:
        """CMB radiation energy density [J/m³]."""
        return 4.0 * SIGMA_SB * self.t_cmb**4 / C