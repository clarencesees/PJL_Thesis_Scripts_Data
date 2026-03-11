"""
Performs a scan over Lambda detuning (relative frequency detuning between
hyperfine components) for Ba-137.

Converted from Ba137_SPDZeeman_Levels_OpticalBloch_Sim_LambdaDetuneScan.m
"""
import numpy as np
from ba137_spd_zeeman_levels_optical_bloch_init_prototype import ba137_init
from ba137_spd_zeeman_levels_optical_bloch_delta_scan import ba137_delta_scan


if __name__ == "__main__":
    Lambda_detune_large_array = np.arange(10, 101, 1)  # Relative frequency detuning scan array

    coeffs = ba137_init()

    # Initialize Rabi frequencies (MHz)
    Omega_S1P2 = 100
    Omega_S2P2 = 100
    Omega_D0P1 = 100
    Omega_D1P1 = 100
    Omega_D2P2 = 100
    Omega_D3P2 = 100

    theta = 0  # Polarization
    Lambda_detune = 1

    Delta_S_array = np.arange(-100, 1, 10)
    Delta_D_array = np.arange(-200, 201, 10)

    # Initial frequency scan to get array dimensions
    sigma_PP_end = ba137_delta_scan(
        coeffs, Delta_S_array, Delta_D_array,
        Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
        theta, theta, Lambda_detune)

    # Initialize arrays for storing results
    sigma_PP_end_detune_large_array = np.full(
        (sigma_PP_end.shape[0], sigma_PP_end.shape[1], len(Lambda_detune_large_array)), np.nan)
    sigma_maxPP_detune_large_array = np.full(len(Lambda_detune_large_array), np.nan)

    # Scan relative frequency detuning
    for Lambda_detune_counter in range(len(Lambda_detune_large_array)):
        ld = Lambda_detune_large_array[Lambda_detune_counter]

        sigma_PP_end = ba137_delta_scan(
            coeffs, Delta_S_array, Delta_D_array,
            Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
            theta, theta, ld)

        sigma_PP_end_detune_large_array[:, :, Lambda_detune_counter] = sigma_PP_end
        sigma_maxPP_detune_large_array[Lambda_detune_counter] = np.nanmax(sigma_PP_end)

        print(f"Lambda_detune={ld:.0f}, max P pop={sigma_maxPP_detune_large_array[Lambda_detune_counter]:.6f}")
