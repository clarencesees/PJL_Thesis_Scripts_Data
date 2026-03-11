import numpy as np


def ba_4levels_shelving_linsolve(Omega_SP, Omega_D1P, Omega_SD2, Delta_SP, Delta_D1P, Delta_SD2=0):
    """
    Calculates steady-state solution of S_1/2, P_1/2, D_3/2 and D_5/2 state populations.

    Parameters:
        Omega_SP: 493nm Rabi frequency (rad*MHz)
        Omega_D1P: 650nm Rabi frequency (rad*MHz)
        Omega_SD2: 1762nm Rabi frequency (rad*MHz)
        Delta_SP: 493nm detuning (rad*MHz)
        Delta_D1P: 650nm detuning (rad*MHz)
        Delta_SD2: 1762nm detuning (rad*MHz)

    Returns:
        sigma_end: Steady-state density matrix elements
        cp1, cp1_conj: P state index arrays
    """
    gamma_S = 95.3  # Decay rate of P_1/2 to S_1/2 in MHz
    gamma_D = 31    # Decay rate of P_1/2 to D_3/2 in MHz

    n = 4  # s1, d1, d2, p1
    # d1 refers ti D_3/2, d2 refers to D_5/2
    # s1 refers to S_1/2, p1 refers to P_1/2
    
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
    cs1_prime = (-1j * Delta_SP * cs1
                 - 1j * (Omega_SP / 2) * cp1
                 - 1j * (Omega_SD2 / 2) * cd2)
    cs1_prime_c = np.conj(-1j * Delta_SP * cs1_conj
                           - 1j * (Omega_SP / 2) * cp1_conj
                           - 1j * (Omega_SD2 / 2) * cd2_conj)

    cd1_prime = (-1j * Delta_D1P * cd1
                 - 1j * (Omega_D1P / 2) * cp1)
    cd1_prime_c = np.conj(-1j * Delta_D1P * cd1_conj
                           - 1j * (Omega_D1P / 2) * cp1_conj)

    cd2_prime = (-1j * (Delta_SP - Delta_SD2) * cd2
                 - 1j * (Omega_SD2 / 2) * cs1)
    cd2_prime_c = np.conj(-1j * (Delta_SP - Delta_SD2) * cd2_conj
                           - 1j * (Omega_SD2 / 2) * cs1_conj)

    cp1_prime = (-1j * (Omega_SP / 2) * cs1
                 - 1j * (Omega_D1P / 2) * cd1)
    cp1_prime_c = np.conj(-1j * (Omega_SP / 2) * cs1_conj
                           - 1j * (Omega_D1P / 2) * cd1_conj)

    c_prime_array = np.array([cs1_prime, cd1_prime, cd2_prime, cp1_prime])
    c_prime_conj_array = np.array([cs1_prime_c, cd1_prime_c, cd2_prime_c, cp1_prime_c])

    ndm = n * n
    Evol_Matrix = np.full((ndm, ndm), np.nan, dtype=complex)

    # Construct evolution matrix
    for i in range(n):
        for j in range(n):
            Pcount = 0
            if i == 3:  # P state
                Pcount += 1
            if j == 3:
                Pcount += 1
            mask = c_array[i, :] * c_conj_array[j, :] == 1
            Evol_Matrix[mask, :] = (c_array[i, :] * c_prime_conj_array[j, :]
                                     + c_prime_array[i, :] * c_conj_array[j, :]
                                     - 0.5 * Pcount * (gamma_S + gamma_D) * c_array[i, :] * c_conj_array[j, :])

    # Override diagonal population equations with decay terms
    Evol_Matrix[cs1 * cs1_conj == 1, :] = (cs1 * cs1_prime_c + cs1_prime * cs1_conj
                                             + gamma_S * cp1 * cp1_conj)
    Evol_Matrix[cd1 * cd1_conj == 1, :] = (cd1 * cd1_prime_c + cd1_prime * cd1_conj
                                             + gamma_D * cp1 * cp1_conj)
    Evol_Matrix[cd2 * cd2_conj == 1, :] = cd2 * cd2_prime_c + cd2_prime * cd2_conj

    # Trace constraint
    trace_row = np.zeros((1, ndm), dtype=complex)
    for i in range(n):
        trace_row[0, c_array[i, :] * c_conj_array[i, :] == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    b = np.zeros(ndm + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end, cp1, cp1_conj


if __name__ == "__main__":
    Omega_SP = 2 * np.pi * 10
    Omega_D1P = 2 * np.pi * 10
    Omega_SD2 = 2 * np.pi * 0.1 * 0
    Delta_SP = -2 * np.pi * 10
    Delta_D1P = -2 * np.pi * 40

    sigma_end, cp1, cp1_conj = ba_4levels_shelving_linsolve(
        Omega_SP, Omega_D1P, Omega_SD2, Delta_SP, Delta_D1P)
    print(f"P state population: {sigma_end[cp1 * cp1_conj == 1].real}")
