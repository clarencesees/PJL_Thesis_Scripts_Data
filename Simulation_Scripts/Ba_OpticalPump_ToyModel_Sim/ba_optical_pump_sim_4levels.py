import numpy as np


def ba_optical_pump_sim_4levels(Omega, Delta_SP, sigma_pi_percentage=5, sigma_minus_percentage=0):
    """
    Computes steady-state of a 4-level optical pumping toy model for barium.
    States: S0 (m=-1/2), S1 (m=+1/2), P0 (m=-1/2), P1 (m=+1/2)

    Parameters:
        Omega: Overall Rabi frequency (rad*MHz)
        Delta_SP: S-P detuning (rad*MHz)
        sigma_pi_percentage: Percentage of pi polarization (0-100)
        sigma_minus_percentage: Percentage of sigma-minus polarization (0-100)

    Returns:
        sigma_end: Steady-state density matrix elements
        cs0, cs1, cp0, cp1: Basis state index arrays
        c_array, c_conj_array: Index arrays for density matrix elements
    """
    sigma_plus_percentage = 100 - sigma_pi_percentage - sigma_minus_percentage

    sigma_pi = np.sqrt(sigma_pi_percentage / 100)
    sigma_minus = np.sqrt(sigma_minus_percentage / 100)
    sigma_plus = np.sqrt(sigma_plus_percentage / 100)

    # Rabi frequencies in MHz
    Omega_S0P0 = sigma_pi * Omega
    Omega_S0P1 = sigma_plus * Omega
    Omega_S1P0 = sigma_minus * Omega
    Omega_S1P1 = sigma_pi * Omega

    gamma_S = 95.3  # Decay rate of P_1/2 state to S_1/2 in MHz

    # Define quantum states as vector arrays.
    s0 = np.zeros(4); s0[0] = 1
    s1 = np.zeros(4); s1[1] = 1
    p0 = np.zeros(4); p0[2] = 1
    p1 = np.zeros(4); p1[3] = 1

    cs0 = np.kron(s0, np.ones(4))
    cs0_conj = np.kron(np.ones(4), s0)
    cs1 = np.kron(s1, np.ones(4))
    cs1_conj = np.kron(np.ones(4), s1)
    cp0 = np.kron(p0, np.ones(4))
    cp0_conj = np.kron(np.ones(4), p0)
    cp1 = np.kron(p1, np.ones(4))
    cp1_conj = np.kron(np.ones(4), p1)

    ind_array = np.eye(len(s0))
    n_states = len(s0)
    n_dm = len(cs0)

    c_array = np.full((n_states, n_dm), np.nan)
    c_conj_array = np.full((n_states, n_dm), np.nan)
    for c_array_count in range(n_states):
        c_array[c_array_count, :] = np.kron(ind_array[c_array_count, :], np.ones(n_states))
        c_conj_array[c_array_count, :] = np.kron(np.ones(n_states), ind_array[c_array_count, :])

    # Compute first derivatives of energy state amplitudes.
    cs0_prime = (-1j * Delta_SP * cs0
                 - 1j * (Omega_S0P0 / 2) * cp0
                 - 1j * (Omega_S0P1 / 2) * cp1)
    cs0_prime_conj = np.conj(-1j * Delta_SP * cs0_conj
                              - 1j * (Omega_S0P0 / 2) * cp0_conj
                              - 1j * (Omega_S0P1 / 2) * cp1_conj)

    cs1_prime = (-1j * Delta_SP * cs1
                 - 1j * (Omega_S1P0 / 2) * cp0
                 - 1j * (Omega_S1P1 / 2) * cp1)
    cs1_prime_conj = np.conj(-1j * Delta_SP * cs1_conj
                              - 1j * (Omega_S1P0 / 2) * cp0_conj
                              - 1j * (Omega_S1P1 / 2) * cp1_conj)

    cp0_prime = (-1j * (Omega_S0P0 / 2) * cs0
                 - 1j * (Omega_S1P0 / 2) * cs1)
    cp0_prime_conj = np.conj(-1j * (Omega_S0P0 / 2) * cs0_conj
                              - 1j * (Omega_S1P0 / 2) * cs1_conj)

    cp1_prime = (-1j * (Omega_S0P1 / 2) * cs0
                 - 1j * (Omega_S1P1 / 2) * cs1)
    cp1_prime_conj = np.conj(-1j * (Omega_S0P1 / 2) * cs0_conj
                              - 1j * (Omega_S1P1 / 2) * cs1_conj)

    c_prime_array = np.array([cs0_prime, cs1_prime, cp0_prime, cp1_prime])
    c_prime_conj_array = np.array([cs0_prime_conj, cs1_prime_conj, cp0_prime_conj, cp1_prime_conj])

    Evol_Matrix = np.full((n_dm, n_dm), np.nan, dtype=complex)

    # Construct the matrix for time-evolution.
    # Note: MATLAB code has a bug: `if EM_count1 == 3 || 4` always evaluates to true in MATLAB.
    # We replicate this behavior here.
    for EM_count1 in range(n_states):
        for EM_count2 in range(n_states):
            Pcount = 0
            # Replicating MATLAB bug: `if EM_count1 == 3 || 4` is always True
            Pcount += 1
            Pcount += 1
            mask = c_array[EM_count1, :] * c_conj_array[EM_count2, :] == 1
            Evol_Matrix[mask, :] = (c_array[EM_count1, :] * c_prime_conj_array[EM_count2, :]
                                    + c_prime_array[EM_count1, :] * c_conj_array[EM_count2, :]
                                    - 0.5 * Pcount * gamma_S * c_array[EM_count1, :] * c_conj_array[EM_count2, :])

    Evol_Matrix[cs0 * cs0_conj == 1, :] = (cs0 * cs0_prime_conj + cs0_prime * cs0_conj
                                             + 0.5 * gamma_S * cp0 * cp0_conj + 0.5 * gamma_S * cp1 * cp1_conj)

    Evol_Matrix[cs1 * cs1_conj == 1, :] = (cs1 * cs1_prime_conj + cs1_prime * cs1_conj
                                             + 0.5 * gamma_S * cp0 * cp0_conj + 0.5 * gamma_S * cp1 * cp1_conj)

    # Add trace constraint
    trace_row = np.zeros((1, n_dm), dtype=complex)
    for EM_count in range(n_states):
        trace_row[0, c_array[EM_count, :] * c_conj_array[EM_count, :] == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    # Solve for the steady-state density matrix.
    b = np.zeros(n_dm + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end, cs0, cs0_conj, cs1, cs1_conj, cp0, cp0_conj, cp1, cp1_conj, c_array, c_conj_array


if __name__ == "__main__":
    Omega = 2 * np.pi * 100
    Delta_SP = -2 * np.pi * 1000
    result = ba_optical_pump_sim_4levels(Omega, Delta_SP)
    sigma_end = result[0]
    cs0, cs0_conj = result[1], result[2]
    cs1, cs1_conj = result[3], result[4]
    cp0, cp0_conj = result[5], result[6]

    print(f"S0 population: {sigma_end[cs0 * cs0_conj == 1].real}")
    print(f"S1 population: {sigma_end[cs1 * cs1_conj == 1].real}")
    print(f"P0 population: {sigma_end[cp0 * cp0_conj == 1].real}")
