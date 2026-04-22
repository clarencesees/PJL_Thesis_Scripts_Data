import numpy as np
from ba138_spd_zeeman_levels_optical_bloch_delta_scan import ba138_delta_scan


def ba138_power_scan():
    """
    Scans Rabi frequencies for Ba-138 Zeeman levels and records P state populations.
    """
    Omega_S_array = np.arange(2, 67, 4)  # S-P Rabi frequencies scan array in MHz
    Omega_D_array = np.arange(2, 63, 4)  # D-P Rabi frequencies scan array in MHz

    Omega_S = 50
    Omega_D = 50
    theta = np.pi / 4

    # Initial scan to get array sizes
    Delta_S_arr, Delta_D_arr, sigma_PP_init = ba138_delta_scan(
        2 * np.pi * Omega_S, 2 * np.pi * Omega_D, theta, theta)

    n_ds = len(Delta_S_arr)
    n_dd = len(Delta_D_arr)
    n_os = len(Omega_S_array)
    n_od = len(Omega_D_array)

    sigma_PP_end_O_array = np.full((n_ds, n_dd, n_os, n_od), np.nan)
    sigma_maxPP_O_array = np.full((n_os, n_od), np.nan)

    for os_idx in range(n_os):
        for od_idx in range(n_od):
            Omega_S_val = 2 * np.pi * Omega_S_array[os_idx]
            Omega_D_val = 2 * np.pi * Omega_D_array[od_idx]
            _, _, sigma_PP_end = ba138_delta_scan(
                Omega_S_val, Omega_D_val, theta, theta)
            sigma_PP_end_O_array[:, :, os_idx, od_idx] = sigma_PP_end
            sigma_maxPP_O_array[os_idx, od_idx] = np.nanmax(sigma_PP_end)
            print(f"Omega_S={Omega_S_array[os_idx]:.0f}, Omega_D={Omega_D_array[od_idx]:.0f}, "
                  f"max P pop={sigma_maxPP_O_array[os_idx, od_idx]:.4f}")

    return Omega_S_array, Omega_D_array, sigma_PP_end_O_array, sigma_maxPP_O_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Omega_S_array, Omega_D_array, _, sigma_maxPP = ba138_power_scan()

    plt.figure()
    plt.pcolormesh(Omega_D_array, Omega_S_array, sigma_maxPP, shading='auto')
    plt.colorbar(label='Max P state population')
    plt.xlabel('Omega_D (MHz)')
    plt.ylabel('Omega_S (MHz)')
    plt.title('Ba-138 Max P state population vs Rabi frequencies')
    plt.show()
