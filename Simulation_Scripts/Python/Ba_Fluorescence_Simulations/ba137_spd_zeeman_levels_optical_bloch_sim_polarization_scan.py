"""
Performs a 2D polarization scan for Ba-137 by running a frequency scan
at each polarization setting and recording the P state populations.

Converted from Ba137_SPDZeeman_Levels_OpticalBloch_Sim_PolarizationScan.m
"""
import numpy as np
from ba137_spd_zeeman_levels_optical_bloch_init_prototype import ba137_init
from ba137_spd_zeeman_levels_optical_bloch_delta_scan import ba137_delta_scan


if __name__ == "__main__":
    theta_S_array = np.arange(0, 181, 10)  # S-P laser polarization scan array (degrees)
    theta_D_array = np.arange(0, 181, 10)  # D-P laser polarization scan array (degrees)

    coeffs = ba137_init()

    # Initialize Rabi frequencies (rad*MHz)
    Omega_S1P2 = 2 * np.pi * 20
    Omega_S2P2 = 2 * np.pi * 20
    Omega_D0P1 = 2 * np.pi * 15
    Omega_D1P1 = 2 * np.pi * 15
    Omega_D2P2 = 2 * np.pi * 15
    Omega_D3P2 = 2 * np.pi * 15

    theta_s = 0
    theta_d = 0
    Lambda_detune = 10

    Delta_S_array = np.arange(-100, 1, 10)
    Delta_D_array = np.arange(-200, 201, 10)

    # Initial frequency scan to get array dimensions
    sigma_PP_end = ba137_delta_scan(
        coeffs, Delta_S_array, Delta_D_array,
        Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
        theta_s, theta_d, Lambda_detune)

    # Initialize arrays for storing results
    sigma_PP_end_theta_array = np.full(
        (sigma_PP_end.shape[0], sigma_PP_end.shape[1], len(theta_S_array), len(theta_D_array)), np.nan)
    sigma_maxPP_theta_array = np.full((len(theta_S_array), len(theta_D_array)), np.nan)

    # Scan laser polarizations
    for theta_S_counter in range(len(theta_S_array)):
        for theta_D_counter in range(len(theta_D_array)):
            ts = (np.pi / 180) * theta_S_array[theta_S_counter]
            td = (np.pi / 180) * theta_D_array[theta_D_counter]

            sigma_PP_end = ba137_delta_scan(
                coeffs, Delta_S_array, Delta_D_array,
                Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
                ts, td, Lambda_detune)

            sigma_PP_end_theta_array[:, :, theta_S_counter, theta_D_counter] = sigma_PP_end
            sigma_maxPP_theta_array[theta_S_counter, theta_D_counter] = np.nanmax(sigma_PP_end)

            print(f"theta_S={theta_S_array[theta_S_counter]:.0f}, "
                  f"theta_D={theta_D_array[theta_D_counter]:.0f}, "
                  f"max P pop={sigma_maxPP_theta_array[theta_S_counter, theta_D_counter]:.6f}")
