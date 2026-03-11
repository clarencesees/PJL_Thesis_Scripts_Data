import numpy as np
from ba_spd_levels_optical_bloch_analytical import ba_spd_levels_optical_bloch_analytical


def run_delta_scan(Omega_S, Omega_D):
    """
    Scans over S-P and D-P detunings and computes steady-state P state population.

    Parameters:
        Omega_S: Rabi frequency of S-P transition (rad*MHz)
        Omega_D: Rabi frequency of D-P transition (rad*MHz)

    Returns:
        Delta_S_array: Array of S-P detunings (MHz)
        Delta_D_array: Array of D-P detunings (MHz)
        sigma_PP_end: 2D array of P state populations
    """
    Delta_S_array = np.arange(-100, 1, 1)  # Range of frequency detunings for S-P in MHz
    Delta_D_array = np.arange(-100, 101, 1)  # Range of frequency detunings for D-P in MHz
    sigma_PP_end = np.full((len(Delta_S_array), len(Delta_D_array)), np.nan)

    for hh in range(len(Delta_S_array)):
        for hhh in range(len(Delta_D_array)):
            Delta_S = 2 * np.pi * Delta_S_array[hh]  # Scale frequency by 2*pi
            Delta_D = 2 * np.pi * Delta_D_array[hhh]  # Scale frequency by 2*pi
            sigma_PP = ba_spd_levels_optical_bloch_analytical(Delta_S, Delta_D, Omega_S, Omega_D)
            sigma_PP_end[hh, hhh] = sigma_PP.real

    # Analytical form may have values divided by zero, giving NaN. Replace with zero.
    sigma_PP_end[np.isnan(sigma_PP_end)] = 0

    return Delta_S_array, Delta_D_array, sigma_PP_end


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Omega_S = 2 * np.pi * 10
    Omega_D = 2 * np.pi * 10
    Delta_S_array, Delta_D_array, sigma_PP_end = run_delta_scan(Omega_S, Omega_D)

    plt.figure()
    plt.pcolormesh(Delta_D_array, Delta_S_array, sigma_PP_end, shading='auto')
    plt.colorbar(label='P state population')
    plt.xlabel('Delta_D (MHz)')
    plt.ylabel('Delta_S (MHz)')
    plt.title('P state population vs detunings')
    plt.show()
