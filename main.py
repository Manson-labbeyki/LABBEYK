"""
Main entry point for the Labbeyk simulation.
"""

from src.labbeyk.constants import DISTANCE_MPC, SimulationParameters
from src.labbeyk.projectile import RelativisticProjectile
from src.labbeyk.kinematics import RelativisticKinematics


def print_case(name, initial_energy, final_energy):
    fraction = final_energy / initial_energy
    lost = 1.0 - fraction

    print(name)
    print("-" * 60)
    print(f"Final total energy : {final_energy:.4e} J")
    print(f"Remaining energy   : {fraction * 100:.8f} %")
    print(f"Energy lost        : {lost * 100:.8f} %")
    print()


def main():
    params = SimulationParameters()
    projectile = RelativisticProjectile(params)
    kin = RelativisticKinematics(params.mass, params.beta0)

    print("=" * 60)
    print("  Labbeyk – Relativistic Projectile Simulation")
    print("=" * 60)

    print(f"Mass                : {params.mass:.1f} kg")
    print(f"Initial beta        : {params.beta0:.6f}")
    print(f"Initial gamma       : {projectile.initial_gamma:.4e}")
    print(f"Travel distance     : {DISTANCE_MPC:.1f} Mpc")
    print(f"CMB cross-section   : {params.cross_section:.4e} m²")
    print(f"IGM density         : {params.n_igm:.4e} m⁻³")
    print(f"IGM cross-section   : {params.igm_cross_section:.4e} m²")
    print(f"IGM efficiency      : {params.igm_efficiency:.4f}")

    print("-" * 60)

    print(f"Initial total energy : {kin.total_energy:.4e} J")
    print(f"Initial kinetic energy: {kin.kinetic_energy:.4e} J")
    print(f"Initial CMB drag     : {projectile.initial_cmb_drag():.4e} N")

    print("=" * 60)

    print("Running three simulations...")
    print()

    _, energy_igm = projectile.simulate_igm_only()
    _, energy_cmb = projectile.simulate_cmb_only()
    _, energy_combined = projectile.simulate()

    print_case(
        "IGM ONLY",
        projectile.initial_energy,
        energy_igm[-1],
    )

    print_case(
        "CMB ONLY",
        projectile.initial_energy,
        energy_cmb[-1],
    )

    print_case(
        "CMB + IGM",
        projectile.initial_energy,
        energy_combined[-1],
    )

    print("=" * 60)
    print("Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()