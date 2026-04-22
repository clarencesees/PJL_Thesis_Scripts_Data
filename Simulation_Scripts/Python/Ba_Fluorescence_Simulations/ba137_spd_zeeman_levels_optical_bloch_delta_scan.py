"""
Performs a 2D frequency scan (cooling and repump detunings) for Ba-137
using the 32-state optical Bloch equation solver.

Converted from Ba137_SPDZeeman_Levels_OpticalBloch_DeltaScan.m
"""
import numpy as np
from ba137_spd_zeeman_levels_optical_bloch_init_prototype import ba137_init
from ba137_spd_zeeman_levels_optical_bloch_linsolve_prototype import ba137_linsolve


def ba137_delta_scan(coeffs, Delta_S_array, Delta_D_array,
                     Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
                     theta_s, theta_d, Lambda_detune):
    """
    Perform a 2D frequency scan over cooling (S) and repump (D) detunings.

    Parameters:
        coeffs: dict from ba137_init()
        Delta_S_array: 1D array of cooling frequency detunings (MHz)
        Delta_D_array: 1D array of repump frequency detunings (MHz)
        Omega_S1P2, Omega_S2P2: S-P Rabi frequencies (rad*MHz)
        Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2: D-P Rabi frequencies (rad*MHz)
        theta_s, theta_d: Polarization angles (rad)
        Lambda_detune: Relative frequency detuning between hyperfine components (MHz)

    Returns:
        sigma_PP_end: 2D array of total P state population (len(Delta_S_array) x len(Delta_D_array))
    """
    sigma_PP_end = np.full((len(Delta_S_array), len(Delta_D_array)), np.nan)

    for hh in range(len(Delta_S_array)):
        for hhh in range(len(Delta_D_array)):
            Delta_S1P2 = 2 * np.pi * Delta_S_array[hh]
            Delta_S2P2 = 2 * np.pi * (Delta_S_array[hh] + Lambda_detune)
            Delta_D0P1 = 2 * np.pi * Delta_D_array[hhh]
            Delta_D1P1 = 2 * np.pi * (Delta_D_array[hhh] + Lambda_detune)
            Delta_D2P2 = 2 * np.pi * Delta_D_array[hhh]
            Delta_D3P2 = 2 * np.pi * (Delta_D_array[hhh] + Lambda_detune)

            sigma_end, c_array, c_conj_array, s1_0 = ba137_linsolve(
                coeffs, theta_s, theta_d,
                Omega_S1P2=Omega_S1P2, Omega_S2P2=Omega_S2P2,
                Omega_D0P1=Omega_D0P1, Omega_D1P1=Omega_D1P1,
                Omega_D2P2=Omega_D2P2, Omega_D3P2=Omega_D3P2,
                Delta_S1P2=Delta_S1P2, Delta_S2P2=Delta_S2P2,
                Delta_D0P1=Delta_D0P1, Delta_D1P1=Delta_D1P1,
                Delta_D2P2=Delta_D2P2, Delta_D3P2=Delta_D3P2)

            # Extract diagonal elements and sum P state populations (indices 24-31)
            n = 32
            CC = np.full(n, np.nan)
            for i in range(n):
                CC[i] = sigma_end[c_array[i, :] * c_conj_array[i, :] == 1].real
            sigma_PP_end[hh, hhh] = np.sum(CC[24:])  # Sum over P states

    return sigma_PP_end


if __name__ == "__main__":
    coeffs = ba137_init()

    Delta_S_array = np.arange(-100, 1, 10)
    Delta_D_array = np.arange(-200, 201, 10)

    Omega_S1P2 = 2 * np.pi * 10
    Omega_S2P2 = 2 * np.pi * 10
    Omega_D0P1 = 2 * np.pi * 10
    Omega_D1P1 = 2 * np.pi * 10
    Omega_D2P2 = 2 * np.pi * 10
    Omega_D3P2 = 2 * np.pi * 10
    theta_s = 0
    theta_d = 0
    Lambda_detune = 10

    sigma_PP_end = ba137_delta_scan(
        coeffs, Delta_S_array, Delta_D_array,
        Omega_S1P2, Omega_S2P2, Omega_D0P1, Omega_D1P1, Omega_D2P2, Omega_D3P2,
        theta_s, theta_d, Lambda_detune)

    print(f"Max P population: {np.nanmax(sigma_PP_end):.6f}")
