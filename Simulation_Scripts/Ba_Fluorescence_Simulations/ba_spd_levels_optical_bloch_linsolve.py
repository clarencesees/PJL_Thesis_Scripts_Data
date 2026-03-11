import numpy as np


def ba_spd_levels_optical_bloch_linsolve(Delta_S, Delta_D, Omega_S, Omega_D):
    """
    Solves the optical Bloch equations for the Ba SPD 3-level system
    using linear solve (steady-state).

    Parameters:
        Delta_S: Detuning of S-P transition laser (rad*MHz)
        Delta_D: Detuning of D-P transition laser (rad*MHz)
        Omega_S: Rabi frequency of S-P transition (rad*MHz)
        Omega_D: Rabi frequency of D-P transition (rad*MHz)

    Returns:
        sigma_end: Steady-state density matrix elements (1D array of length 9)
    """
    gamma_S = 95.3  # Decay rate of P_1/2 state to S_1/2 in MHz
    gamma_D = 31    # Decay rate of P_1/2 state to D_3/2 in MHz

    # Define basis states as vectors.
    cs0 = np.kron([1, 0, 0], np.ones(3))
    cs0_conj = np.kron(np.ones(3), [1, 0, 0])
    cd0 = np.kron([0, 1, 0], np.ones(3))
    cd0_conj = np.kron(np.ones(3), [0, 1, 0])
    cp0 = np.kron([0, 0, 1], np.ones(3))
    cp0_conj = np.kron(np.ones(3), [0, 0, 1])

    # Define the time derivatives of each state probability amplitude.
    cs0_prime = -1j * Delta_S * cs0 - 1j * (Omega_S / 2) * cp0
    cs0_prime_conj = np.conj(-1j * Delta_S * cs0_conj - 1j * (Omega_S / 2) * cp0_conj)

    cd0_prime = -1j * Delta_D * cd0 - 1j * (Omega_D / 2) * cp0
    cd0_prime_conj = np.conj(-1j * Delta_D * cd0_conj - 1j * (Omega_D / 2) * cp0_conj)

    cp0_prime = -1j * (Omega_S / 2) * cs0 - 1j * (Omega_D / 2) * cd0
    cp0_prime_conj = np.conj(-1j * (Omega_S / 2) * cs0_conj - 1j * (Omega_D / 2) * cd0_conj)

    n = len(cs0)
    Evol_Matrix = np.full((n, n), np.nan, dtype=complex)

    # Construct the time-evolution matrix for the density matrix.
    Evol_Matrix[cs0 * cs0_conj == 1, :] = (cs0 * cs0_prime_conj + cs0_prime * cs0_conj
                                             + gamma_S * cp0 * cp0_conj)

    Evol_Matrix[cd0 * cd0_conj == 1, :] = (cd0 * cd0_prime_conj + cd0_prime * cd0_conj
                                             + gamma_D * cp0 * cp0_conj)

    Evol_Matrix[cp0 * cp0_conj == 1, :] = (cp0 * cp0_prime_conj + cp0_prime * cp0_conj
                                             - (gamma_S + gamma_D) * cp0 * cp0_conj)

    Evol_Matrix[cs0 * cd0_conj == 1, :] = cs0 * cd0_prime_conj + cs0_prime * cd0_conj
    Evol_Matrix[cd0 * cs0_conj == 1, :] = cd0 * cs0_prime_conj + cd0_prime * cs0_conj

    Evol_Matrix[cs0 * cp0_conj == 1, :] = (cs0 * cp0_prime_conj + cs0_prime * cp0_conj
                                             - ((gamma_S + gamma_D) / 2) * cs0 * cp0_conj)
    Evol_Matrix[cp0 * cs0_conj == 1, :] = (cp0 * cs0_prime_conj + cp0_prime * cs0_conj
                                             - ((gamma_S + gamma_D) / 2) * cp0 * cs0_conj)

    Evol_Matrix[cd0 * cp0_conj == 1, :] = (cd0 * cp0_prime_conj + cd0_prime * cp0_conj
                                             - ((gamma_S + gamma_D) / 2) * cd0 * cp0_conj)
    Evol_Matrix[cp0 * cd0_conj == 1, :] = (cp0 * cd0_prime_conj + cp0_prime * cd0_conj
                                             - ((gamma_S + gamma_D) / 2) * cp0 * cd0_conj)

    # Add trace constraint: trace of density matrix = 1
    trace_row = np.zeros((1, n), dtype=complex)
    trace_row[0, cs0 * cs0_conj == 1] = 1
    trace_row[0, cp0 * cp0_conj == 1] = 1
    trace_row[0, cd0 * cd0_conj == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    # Solve for the steady-state density matrix.
    b = np.zeros(n + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end


if __name__ == "__main__":
    Delta_S = -2 * np.pi * 10
    Delta_D = -2 * np.pi * 40
    Omega_S = 2 * np.pi * 10
    Omega_D = 2 * np.pi * 10
    sigma = ba_spd_levels_optical_bloch_linsolve(Delta_S, Delta_D, Omega_S, Omega_D)
    print("Steady-state density matrix elements:")
    print(sigma)
