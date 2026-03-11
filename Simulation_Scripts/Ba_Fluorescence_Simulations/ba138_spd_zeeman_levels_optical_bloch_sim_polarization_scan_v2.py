import numpy as np
from ba138_spd_zeeman_levels_optical_bloch_delta_scan import ba138_delta_scan


def ba138_polarization_scan():
    """
    Scans laser polarizations for Ba-138 Zeeman levels and records P state populations.
    """
    theta_S_array = np.arange(0, 181, 10)  # S-P laser polarization scan array
    theta_D_array = np.arange(0, 181, 10)  # D-P laser polarization scan array

    Omega_S = 2 * np.pi * 15
    Omega_D = 2 * np.pi * 25
    theta_s = 0
    theta_d = 0

    # Initial scan to get array sizes
    Delta_S_arr, Delta_D_arr, sigma_PP_init = ba138_delta_scan(
        Omega_S, Omega_D, theta_s, theta_d)

    n_ds = len(Delta_S_arr)
    n_dd = len(Delta_D_arr)
    n_ts = len(theta_S_array)
    n_td = len(theta_D_array)

    sigma_PP_end_theta_array = np.full((n_ds, n_dd, n_ts, n_td), np.nan)
    sigma_maxPP_theta_array = np.full((n_ts, n_td), np.nan)

    for ts_idx in range(n_ts):
        for td_idx in range(n_td):
            theta_s_val = (np.pi / 180) * theta_S_array[ts_idx]
            theta_d_val = (np.pi / 180) * theta_D_array[td_idx]
            _, _, sigma_PP_end = ba138_delta_scan(
                Omega_S, Omega_D, theta_s_val, theta_d_val)
            sigma_PP_end_theta_array[:, :, ts_idx, td_idx] = sigma_PP_end
            sigma_maxPP_theta_array[ts_idx, td_idx] = np.nanmax(sigma_PP_end)
            print(f"theta_S={theta_S_array[ts_idx]:.0f}, theta_D={theta_D_array[td_idx]:.0f}, "
                  f"max P pop={sigma_maxPP_theta_array[ts_idx, td_idx]:.4f}")

    return theta_S_array, theta_D_array, sigma_PP_end_theta_array, sigma_maxPP_theta_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    theta_S_array, theta_D_array, _, sigma_maxPP = ba138_polarization_scan()

    plt.figure()
    plt.pcolormesh(theta_D_array, theta_S_array, sigma_maxPP, shading='auto')
    plt.colorbar(label='Max P state population')
    plt.xlabel('theta_D (degrees)')
    plt.ylabel('theta_S (degrees)')
    plt.title('Ba-138 Max P state population vs polarizations')
    plt.show()
