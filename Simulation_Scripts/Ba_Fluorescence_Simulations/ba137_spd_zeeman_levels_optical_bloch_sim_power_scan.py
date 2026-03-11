"""
Performs a 2D power (Rabi frequency) scan for Ba-137 by running a frequency
scan at each power setting and recording the P state populations.

Converted from Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PowerScan.m
"""
import numpy as np
from ba137_spd_zeeman_levels_optical_bloch_init_prototype import ba137_init
from ba137_spd_zeeman_levels_optical_bloch_delta_scan import ba137_delta_scan


if __name__ == "__main__":
    Omega_S_array = np.arange(2, 67, 4)  # S-P Rabi frequencies scan array (MHz)
    Omega_D_array = np.arange(2, 63, 4)  # D-P Rabi frequencies scan array (MHz)

    coeffs = ba137_init()

    # Initial Rabi frequencies (MHz)
    Omega_S1P2 = 50
    Omega_S2P2 = 50
    Omega_D0P1 = 50
    Omega_D1P1 = 50
    Omega_D2P2 = 50
    Omega_D3P2 = 50

    theta = np.pi / 4  # Polarization
    Lambda_detune = 10  # Relative frequency detuning (MHz)

    Delta_S_array = np.arange(-100, 1, 10)
    Delta_D_array = np.arange(-200, 201, 10)

    # Initial frequency scan to get array dimensions
    sigma_PP_end = ba137_delta_scan(
        coeffs, Delta_S_array, Delta_D_array,
        2 * np.pi * Omega_S1P2, 2 * np.pi * Omega_S2P2,
        2 * np.pi * Omega_D0P1, 2 * np.pi * Omega_D1P1,
        2 * np.pi * Omega_D2P2, 2 * np.pi * Omega_D3P2,
        theta, theta, Lambda_detune)

    # Initialize arrays for storing results
    sigma_PP_end_O_array = np.full(
        (sigma_PP_end.shape[0], sigma_PP_end.shape[1], len(Omega_S_array), len(Omega_D_array)), np.nan)
    sigma_maxPP_O_array = np.full((len(Omega_S_array), len(Omega_D_array)), np.nan)

    # Scan Rabi frequencies
    for Omega_S_counter in range(len(Omega_S_array)):
        for Omega_D_counter in range(len(Omega_D_array)):
            Os = 2 * np.pi * Omega_S_array[Omega_S_counter]
            Od = 2 * np.pi * Omega_D_array[Omega_D_counter]

            sigma_PP_end = ba137_delta_scan(
                coeffs, Delta_S_array, Delta_D_array,
                Os, Os, Od, Od, Od, Od,
                theta, theta, Lambda_detune)

            sigma_PP_end_O_array[:, :, Omega_S_counter, Omega_D_counter] = sigma_PP_end
            sigma_maxPP_O_array[Omega_S_counter, Omega_D_counter] = np.nanmax(sigma_PP_end)

            print(f"Omega_S={Omega_S_array[Omega_S_counter]:.0f}, "
                  f"Omega_D={Omega_D_array[Omega_D_counter]:.0f}, "
                  f"max P pop={sigma_maxPP_O_array[Omega_S_counter, Omega_D_counter]:.6f}")
