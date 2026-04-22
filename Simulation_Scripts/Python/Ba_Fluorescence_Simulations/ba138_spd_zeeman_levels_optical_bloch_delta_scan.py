import numpy as np
from ba138_spd_zeeman_levels_optical_bloch_linsolve_v2 import ba138_spd_zeeman_linsolve_v2


def ba138_delta_scan(Omega_S, Omega_D, theta_s, theta_d, B_gauss, Delta_S_array=None, Delta_D_array=None):
    """
    Scans over S-P and D-P detunings for Ba-138 Zeeman levels.

    Parameters:
        Omega_S: S-P Rabi frequency (rad*MHz)
        Omega_D: D-P Rabi frequency (rad*MHz)
        theta_s: S-P laser polarization angle (rad)
        theta_d: D-P laser polarization angle (rad)
        B_gauss: Magnetic field magnitude (Gauss)
        Delta_S_array: Array of S-P detunings in MHz (default: -100:10:0)
        Delta_D_array: Array of D-P detunings in MHz (default: -100:10:100)

    Returns:
        Delta_S_array, Delta_D_array, sigma_PP_end
    """
    if Delta_S_array is None:
        Delta_S_array = np.arange(-100, 1, 10)
    if Delta_D_array is None:
        Delta_D_array = np.arange(-100, 101, 10)

    sigma_PP_end = np.full((len(Delta_S_array), len(Delta_D_array)), np.nan)

    for hh in range(len(Delta_S_array)):
        for hhh in range(len(Delta_D_array)):
            Delta_SP01 = 2 * np.pi * Delta_S_array[hh]
            Delta_DP01 = 2 * np.pi * Delta_D_array[hhh]
            sigma_end, cp0, cp0_c, cp1, cp1_c = ba138_spd_zeeman_linsolve_v2(
                Delta_SP01, Delta_DP01, Omega_S, Omega_D, theta_s, theta_d, B_gauss)
            sigma_PP_end[hh, hhh] = (sigma_end[cp0 * cp0_c == 1].real
                                      + sigma_end[cp1 * cp1_c == 1].real)

    return Delta_S_array, Delta_D_array, sigma_PP_end


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Omega_S = 2 * np.pi * 15
    Omega_D = 2 * np.pi * 25
    theta_s = 0
    theta_d = 0
    B_gauss = 8.35

    Delta_S_array, Delta_D_array, sigma_PP_end = ba138_delta_scan(
        Omega_S, Omega_D, theta_s, theta_d, B_gauss)

    plt.figure()
    plt.pcolormesh(Delta_D_array, Delta_S_array, sigma_PP_end, shading='auto')
    plt.colorbar(label='P state population')
    plt.xlabel('Delta_D (MHz)')
    plt.ylabel('Delta_S (MHz)')
    plt.title('Ba-138 P state population vs detunings')
    plt.show()
