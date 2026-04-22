import numpy as np
from math import sqrt, cos, sin


def ba138_spd_zeeman_linsolve(Delta_SP01, Delta_DP01, Omega_S, Omega_D, theta):
    """
    Solves the optical Bloch equations for Ba-138 SPD Zeeman levels (8 states)
    using linear solve (steady-state).

    Parameters:
        Delta_SP01: S-P detuning (rad*MHz)
        Delta_DP01: D-P detuning (rad*MHz)
        Omega_S: S-P Rabi frequency (rad*MHz)
        Omega_D: D-P Rabi frequency (rad*MHz)
        theta: Polarization angle (rad)

    Returns:
        sigma_end: Steady-state density matrix elements
        State index arrays for extracting specific populations
    """
    Planck_h = 6.62607015e-34
    uB = 9.274009994e-24
    B = 1 * 470e-6
    Delta_zs = 1e-6 * 2 * uB * B / Planck_h
    Delta_zd = 1e-6 * (4 / 5) * uB * B / Planck_h
    Delta_zp = 1e-6 * (2 / 3) * uB * B / Planck_h

    pol_minus_s = sqrt((cos(theta)**2) / 2)
    pol_0_s = sqrt(sin(theta)**2)
    pol_plus_s = sqrt((cos(theta)**2) / 2)
    pol_minus_d = sqrt((cos(theta)**2) / 2)
    pol_0_d = sqrt(sin(theta)**2)
    pol_plus_d = sqrt((cos(theta)**2) / 2)

    gamma_S = 95.3
    gamma_D = 31

    # Define energy states (8 states: s0, s1, d0, d1, d2, d3, p0, p1)
    n = 8
    def make_basis(idx):
        v = np.zeros(n)
        v[idx] = 1
        return v

    states = [make_basis(i) for i in range(n)]
    c = [np.kron(s, np.ones(n)) for s in states]
    c_conj = [np.kron(np.ones(n), s) for s in states]

    cs0, cs1, cd0, cd1, cd2, cd3, cp0, cp1 = c
    cs0_c, cs1_c, cd0_c, cd1_c, cd2_c, cd3_c, cp0_c, cp1_c = c_conj

    # Compute first derivatives
    cs0_prime = (1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cp0 + sqrt(2 / 3) * pol_plus_s * cp1))
    cs0_prime_c = np.conj(1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cp0_c + sqrt(2 / 3) * pol_plus_s * cp1_c))

    cs1_prime = (-1j * Delta_zs * cs1
                 - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cp0 + sqrt(1 / 3) * pol_0_s * cp1))
    cs1_prime_c = np.conj(-1j * Delta_zs * cs1_c
                           - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cp0_c + sqrt(1 / 3) * pol_0_s * cp1_c))

    cd0_prime = (1j * (Delta_SP01 - Delta_DP01) * cd0
                 - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_plus_d * cp0)
    cd0_prime_c = np.conj(1j * (Delta_SP01 - Delta_DP01) * cd0_c
                           - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_plus_d * cp0_c)

    cd1_prime = (1j * (Delta_SP01 - Delta_DP01 - Delta_zd) * cd1
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cp1 - sqrt(1 / 3) * pol_0_d * cp0))
    cd1_prime_c = np.conj(1j * (Delta_SP01 - Delta_DP01 - Delta_zd) * cd1_c
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cp1_c - sqrt(1 / 3) * pol_0_d * cp0_c))

    cd2_prime = (1j * (Delta_SP01 - Delta_DP01 - 2 * Delta_zd) * cd2
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_minus_d * cp0 - sqrt(1 / 3) * pol_0_d * cp1))
    cd2_prime_c = np.conj(1j * (Delta_SP01 - Delta_DP01 - 2 * Delta_zd) * cd2_c
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_minus_d * cp0_c - sqrt(1 / 3) * pol_0_d * cp1_c))

    cd3_prime = (1j * (Delta_SP01 - Delta_DP01 - 3 * Delta_zd) * cd3
                 - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_minus_d * cp1)
    cd3_prime_c = np.conj(1j * (Delta_SP01 - Delta_DP01 - 3 * Delta_zd) * cd3_c
                           - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_minus_d * cp1_c)

    cp0_prime = (1j * (Delta_SP01 + Delta_zp) * cp0
                 - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cs1 - sqrt(1 / 3) * pol_0_s * cs0)
                 - 1j * (Omega_D / 2) * (sqrt(1 / 2) * pol_plus_d * cd0 - sqrt(1 / 3) * pol_0_d * cd1 + sqrt(1 / 6) * pol_minus_d * cd2))
    cp0_prime_c = np.conj(1j * (Delta_SP01 + Delta_zp) * cp0_c
                           - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cs1_c - sqrt(1 / 3) * pol_0_s * cs0_c)
                           - 1j * (Omega_D / 2) * (sqrt(1 / 2) * pol_plus_d * cd0_c - sqrt(1 / 3) * pol_0_d * cd1_c + sqrt(1 / 6) * pol_minus_d * cd2_c))

    cp1_prime = (1j * Delta_SP01 * cp1
                 - 1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cs1 - sqrt(2 / 3) * pol_plus_s * cs0)
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cd1 - sqrt(1 / 3) * pol_0_d * cd2 + sqrt(1 / 2) * pol_minus_d * cd3))
    cp1_prime_c = np.conj(1j * Delta_SP01 * cp1_c
                           - 1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cs1_c - sqrt(2 / 3) * pol_plus_s * cs0_c)
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cd1_c - sqrt(1 / 3) * pol_0_d * cd2_c + sqrt(1 / 2) * pol_minus_d * cd3_c))

    primes = [cs0_prime, cs1_prime, cd0_prime, cd1_prime, cd2_prime, cd3_prime, cp0_prime, cp1_prime]
    primes_c = [cs0_prime_c, cs1_prime_c, cd0_prime_c, cd1_prime_c, cd2_prime_c, cd3_prime_c, cp0_prime_c, cp1_prime_c]

    ndm = n * n
    Evol_Matrix = np.full((ndm, ndm), np.nan, dtype=complex)

    # Diagonal population equations with decay
    Evol_Matrix[cs0 * cs0_c == 1, :] = (cs0 * cs0_prime_c + cs0_prime * cs0_c
                                          + (1 / 3) * gamma_S * cp0 * cp0_c + (2 / 3) * gamma_S * cp1 * cp1_c)
    Evol_Matrix[cs1 * cs1_c == 1, :] = (cs1 * cs1_prime_c + cs1_prime * cs1_c
                                          + (2 / 3) * gamma_S * cp0 * cp0_c + (1 / 3) * gamma_S * cp1 * cp1_c)
    Evol_Matrix[cd0 * cd0_c == 1, :] = (cd0 * cd0_prime_c + cd0_prime * cd0_c
                                          + (1 / 2) * gamma_D * cp0 * cp0_c)
    Evol_Matrix[cd1 * cd1_c == 1, :] = (cd1 * cd1_prime_c + cd1_prime * cd1_c
                                          + (1 / 3) * gamma_D * cp0 * cp0_c + (1 / 6) * gamma_D * cp1 * cp1_c)
    Evol_Matrix[cd2 * cd2_c == 1, :] = (cd2 * cd2_prime_c + cd2_prime * cd2_c
                                          + (1 / 6) * gamma_D * cp0 * cp0_c + (1 / 3) * gamma_D * cp1 * cp1_c)
    Evol_Matrix[cd3 * cd3_c == 1, :] = (cd3 * cd3_prime_c + cd3_prime * cd3_c
                                          + (1 / 2) * gamma_D * cp1 * cp1_c)
    Evol_Matrix[cp0 * cp0_c == 1, :] = (cp0 * cp0_prime_c + cp0_prime * cp0_c
                                          - (gamma_S + gamma_D) * cp0 * cp0_c)
    Evol_Matrix[cp1 * cp1_c == 1, :] = (cp1 * cp1_prime_c + cp1_prime * cp1_c
                                          - (gamma_S + gamma_D) * cp1 * cp1_c)

    # Off-diagonal coherence equations
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((i, j))

    for i, j in pairs:
        mask = c[i] * c_conj[j] == 1
        if not np.any(mask):
            continue
        # Check if already assigned (diagonal populations above)
        if i == j:
            continue
        p_count = 0
        if i >= 6:  # P states are indices 6, 7
            p_count += 1
        if j >= 6:
            p_count += 1
        Evol_Matrix[mask, :] = (c[i] * primes_c[j] + primes[i] * c_conj[j]
                                 - (p_count / 2) * (gamma_S + gamma_D) * c[i] * c_conj[j])

    # Trace constraint
    trace_row = np.zeros((1, ndm), dtype=complex)
    for i in range(n):
        trace_row[0, c[i] * c_conj[i] == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    b = np.zeros(ndm + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end, cp0, cp0_c, cp1, cp1_c


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    theta = np.pi / 4
    Omega_S = 2 * np.pi * 15
    Omega_D = 2 * np.pi * 25
    Delta_SP01 = -2 * np.pi * 20
    Delta_DP01 = -2 * np.pi * 40

    sigma_end, cp0, cp0_c, cp1, cp1_c = ba138_spd_zeeman_linsolve(
        Delta_SP01, Delta_DP01, Omega_S, Omega_D, theta)

    P_pop = sigma_end[cp0 * cp0_c == 1].real + sigma_end[cp1 * cp1_c == 1].real
    print(f"P state population: {P_pop}")
