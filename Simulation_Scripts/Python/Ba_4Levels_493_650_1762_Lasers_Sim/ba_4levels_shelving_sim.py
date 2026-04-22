import numpy as np


def ba_4levels_shelving_sim():
    """
    Time-resolved simulation of S_1/2, P_1/2, D_3/2 and D_5/2 state populations
    using Runge-Kutta method.

    Returns:
        t: Time array
        sigma_t: Time-resolved density matrix elements
        cs1, cd1, cd2, cp1 and their conjugate arrays
    """
    Omega_SP = 2 * np.pi * 10
    Omega_D1P = 0 * 2 * np.pi * 10
    Omega_SD2 = 2 * np.pi * 10
    Delta_SP = -2 * np.pi * 20
    Delta_D1P = -2 * np.pi * 50
    Delta_SD2 = -2 * np.pi * 0

    gamma_S = 95.3
    gamma_D = 31
    gamma_D52 = 2.86e-8

    n = 4
    s1 = np.zeros(n); s1[0] = 1
    d1 = np.zeros(n); d1[1] = 1
    d2 = np.zeros(n); d2[2] = 1
    p1 = np.zeros(n); p1[3] = 1

    cs1 = np.kron(s1, np.ones(n))
    cs1_conj = np.kron(np.ones(n), s1)
    cd1 = np.kron(d1, np.ones(n))
    cd1_conj = np.kron(np.ones(n), d1)
    cd2 = np.kron(d2, np.ones(n))
    cd2_conj = np.kron(np.ones(n), d2)
    cp1 = np.kron(p1, np.ones(n))
    cp1_conj = np.kron(np.ones(n), p1)

    ind_array = np.eye(n)
    c_array = np.full((n, n * n), np.nan)
    c_conj_array = np.full((n, n * n), np.nan)
    for i in range(n):
        c_array[i, :] = np.kron(ind_array[i, :], np.ones(n))
        c_conj_array[i, :] = np.kron(np.ones(n), ind_array[i, :])

    # Time derivatives
    cs1_prime = -1j * Delta_SP * cs1 - 1j * (Omega_SP / 2) * cp1 - 1j * (Omega_SD2 / 2) * cd2
    cs1_prime_c = np.conj(-1j * Delta_SP * cs1_conj - 1j * (Omega_SP / 2) * cp1_conj - 1j * (Omega_SD2 / 2) * cd2_conj)

    cd1_prime = -1j * Delta_D1P * cd1 - 1j * (Omega_D1P / 2) * cp1
    cd1_prime_c = np.conj(-1j * Delta_D1P * cd1_conj - 1j * (Omega_D1P / 2) * cp1_conj)

    cd2_prime = -1j * (Delta_SP - Delta_SD2) * cd2 - 1j * (Omega_SD2 / 2) * cs1
    cd2_prime_c = np.conj(-1j * (Delta_SP - Delta_SD2) * cd2_conj - 1j * (Omega_SD2 / 2) * cs1_conj)

    cp1_prime = -1j * (Omega_SP / 2) * cs1 - 1j * (Omega_D1P / 2) * cd1
    cp1_prime_c = np.conj(-1j * (Omega_SP / 2) * cs1_conj - 1j * (Omega_D1P / 2) * cd1_conj)

    c_prime_array = np.array([cs1_prime, cd1_prime, cd2_prime, cp1_prime])
    c_prime_conj_array = np.array([cs1_prime_c, cd1_prime_c, cd2_prime_c, cp1_prime_c])

    ndm = n * n
    Evol_Matrix = np.full((ndm, ndm), np.nan, dtype=complex)

    for i in range(n):
        for j in range(n):
            Pcount = 0
            D2count = 0
            if i == 3:
                Pcount += 1
            if j == 3:
                Pcount += 1
            if i == 2:
                D2count += 1
            if j == 2:
                D2count += 1
            mask = c_array[i, :] * c_conj_array[j, :] == 1
            Evol_Matrix[mask, :] = (c_array[i, :] * c_prime_conj_array[j, :]
                                     + c_prime_array[i, :] * c_conj_array[j, :]
                                     - 0.5 * Pcount * (gamma_S + gamma_D) * c_array[i, :] * c_conj_array[j, :]
                                     - 0.5 * D2count * gamma_D52 * c_array[i, :] * c_conj_array[j, :])

    Evol_Matrix[cs1 * cs1_conj == 1, :] = (cs1 * cs1_prime_c + cs1_prime * cs1_conj
                                             + gamma_S * cp1 * cp1_conj + gamma_D52 * cd2 * cd2_conj)
    Evol_Matrix[cd1 * cd1_conj == 1, :] = (cd1 * cd1_prime_c + cd1_prime * cd1_conj
                                             + gamma_D * cp1 * cp1_conj)
    Evol_Matrix[cd2 * cd2_conj == 1, :] = (cd2 * cd2_prime_c + cd2_prime * cd2_conj
                                             - gamma_D52 * cd2 * cd2_conj)

    # Time evolution using Runge-Kutta
    deltat = 1e-4
    t_end = 100
    t = np.arange(0, t_end + deltat, deltat)

    sigma_t = np.full((ndm, len(t)), np.nan, dtype=complex)
    sigma_t[:, 0] = 0
    sigma_t[cd2 * cd2_conj == 1, 0] = 1  # Initial state in D_5/2

    for h in range(1, len(t)):
        dt = t[h] - t[h - 1]
        s = sigma_t[:, h - 1]
        k1 = Evol_Matrix @ s * dt
        k2 = Evol_Matrix @ (s + k1 / 2) * dt
        k3 = Evol_Matrix @ (s + k2 / 2) * dt
        k4 = Evol_Matrix @ (s + k3) * dt
        sigma_t[:, h] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, sigma_t, cs1, cs1_conj, cd1, cd1_conj, cd2, cd2_conj, cp1, cp1_conj


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, sigma_t, cs1, cs1_c, cd1, cd1_c, cd2, cd2_c, cp1, cp1_c = ba_4levels_shelving_sim()

    plt.figure()
    plt.plot(t, sigma_t[cs1 * cs1_c == 1, :].real.flatten(), label='S')
    plt.plot(t, sigma_t[cd1 * cd1_c == 1, :].real.flatten(), label='D_3/2')
    plt.plot(t, sigma_t[cd2 * cd2_c == 1, :].real.flatten(), label='D_5/2')
    plt.plot(t, sigma_t[cp1 * cp1_c == 1, :].real.flatten(), label='P')
    plt.xlabel('Time (us)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Ba 4-Level Shelving Simulation')
    plt.show()
