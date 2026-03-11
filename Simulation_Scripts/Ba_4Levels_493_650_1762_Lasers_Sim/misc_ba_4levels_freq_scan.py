import numpy as np
from ba_4levels_shelving_sim_linsolve import ba_4levels_shelving_linsolve


def misc_ba_4levels_freq_scan():
    """
    Frequency scan of 1762nm laser for 4-level Ba system.

    Returns:
        Delta_SD2_array: Array of 1762nm detunings (MHz)
        sigma_end_array: P state population for each detuning
    """
    Delta_SD2_array = np.arange(-1000, 1001, 1)
    sigma_end_array = np.full(len(Delta_SD2_array), np.nan)

    Omega_SP = 2 * np.pi * 10
    Omega_D1P = 2 * np.pi * 10
    Omega_SD2 = 2 * np.pi * 0.1
    Delta_SP = -2 * np.pi * 10
    Delta_D1P = -2 * np.pi * 40

    for idx in range(len(Delta_SD2_array)):
        Delta_SD2 = 2 * np.pi * Delta_SD2_array[idx]
        sigma_end, cp1, cp1_conj = ba_4levels_shelving_linsolve(
            Omega_SP, Omega_D1P, Omega_SD2, Delta_SP, Delta_D1P, Delta_SD2)
        sigma_end_array[idx] = sigma_end[cp1 * cp1_conj == 1].real

    return Delta_SD2_array, sigma_end_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Delta_SD2_array, sigma_end_array = misc_ba_4levels_freq_scan()

    plt.figure()
    plt.plot(Delta_SD2_array, sigma_end_array)
    plt.xlabel('Delta_SD2 (MHz)')
    plt.ylabel('P state population')
    plt.title('Ba 4-Level Frequency Scan (1762 nm)')
    plt.show()
