"""
Computes the density matrix in the equilibrated state from the optical Bloch
equations for Ba-137 with full hyperfine+Zeeman structure (32 states).

This is a direct conversion of the MATLAB script
Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.m.

Requires: ba137_spd_zeeman_levels_optical_bloch_init_prototype.py to be run first
to obtain the Clebsch-Gordan and Wigner-Eckart coefficients.
"""
import numpy as np
from math import sqrt, cos, sin


def ba137_linsolve(coeffs, theta_s, theta_d,
                   Omega_S1P1=0, Omega_S1P2=0, Omega_S2P1=0, Omega_S2P2=0,
                   Omega_D0P1=0, Omega_D1P1=0, Omega_D1P2=0, Omega_D2P1=0, Omega_D2P2=0, Omega_D3P2=0,
                   Delta_S1P1=0, Delta_S1P2=0, Delta_S2P1=0, Delta_S2P2=0,
                   Delta_D0P1=0, Delta_D1P1=0, Delta_D1P2=0, Delta_D2P1=0, Delta_D2P2=0, Delta_D3P2=0):
    """
    Solve steady-state optical Bloch equations for Ba-137 (32 Zeeman states).

    Parameters:
        coeffs: dict from ba137_init() containing CG and WE coefficients
        theta_s: S-P laser polarization angle (rad)
        theta_d: D-P laser polarization angle (rad)
        Omega_*/Delta_*: Rabi frequencies and detunings (rad*MHz)

    Returns:
        sigma_end: Steady-state density matrix elements (1024-element array)
        c_array, c_conj_array: Index arrays for extracting density matrix elements
        s1_0: Reference state array (for size information)
    """
    Planck_h = 6.62607015e-34
    uB = 9.274009994e-24
    B = 1 * 470e-6

    L_S, L_P, L_D = 0, 1, 2
    S_12 = 0.5
    J_12, J_32 = 0.5, 1.5
    F_s1, F_s2 = 1, 2
    F_p1, F_p2 = 1, 2
    F_d0, F_d1, F_d2, F_d3 = 0, 1, 2, 3
    g_S, g_L = 2, 1
    I_spin = 1.5

    # g-factors
    g_J_S = (g_L * (J_12 * (J_12 + 1) - S_12 * (S_12 + 1) + L_S * (L_S + 1)) / (2 * J_12 * (J_12 + 1))
             + g_S * (J_12 * (J_12 + 1) + S_12 * (S_12 + 1) - L_S * (L_S + 1)) / (2 * J_12 * (J_12 + 1)))
    g_J_P = (g_L * (J_12 * (J_12 + 1) - S_12 * (S_12 + 1) + L_P * (L_P + 1)) / (2 * J_12 * (J_12 + 1))
             + g_S * (J_12 * (J_12 + 1) + S_12 * (S_12 + 1) - L_P * (L_P + 1)) / (2 * J_12 * (J_12 + 1)))
    g_J_D = (g_L * (J_32 * (J_32 + 1) - S_12 * (S_12 + 1) + L_D * (L_D + 1)) / (2 * J_32 * (J_32 + 1))
             + g_S * (J_32 * (J_32 + 1) + S_12 * (S_12 + 1) - L_D * (L_D + 1)) / (2 * J_32 * (J_32 + 1)))

    def _gF(gJ, F, J):
        if F == 0:
            return 0.0
        return gJ * (F * (F + 1) - I_spin * (I_spin + 1) + J * (J + 1)) / (2 * F * (F + 1))

    g_F_s1 = _gF(g_J_S, F_s1, J_12)
    g_F_s2 = _gF(g_J_S, F_s2, J_12)
    g_F_p1 = _gF(g_J_P, F_p1, J_12)
    g_F_p2 = _gF(g_J_P, F_p2, J_12)
    g_F_d1 = _gF(g_J_D, F_d1, J_32)
    g_F_d2 = _gF(g_J_D, F_d2, J_32)
    g_F_d3 = _gF(g_J_D, F_d3, J_32)

    # Zeeman splittings (MHz)
    Delta_zs1 = 1e-6 * g_F_s1 * uB * B / Planck_h
    Delta_zs2 = 1e-6 * g_F_s2 * uB * B / Planck_h
    Delta_zd1 = 1e-6 * g_F_d1 * uB * B / Planck_h
    Delta_zd2 = 1e-6 * g_F_d2 * uB * B / Planck_h
    Delta_zd3 = 1e-6 * g_F_d3 * uB * B / Planck_h
    Delta_zp1 = 1e-6 * g_F_p1 * uB * B / Planck_h
    Delta_zp2 = 1e-6 * g_F_p2 * uB * B / Planck_h

    # Polarization components
    pol_sp_m = sqrt((cos(theta_s)**2) / 2)
    pol_sp_0 = sqrt(sin(theta_s)**2)
    pol_sp_p = sqrt((cos(theta_s)**2) / 2)
    pol_dp_m = sqrt((cos(theta_d)**2) / 2)
    pol_dp_0 = sqrt(sin(theta_d)**2)
    pol_dp_p = sqrt((cos(theta_d)**2) / 2)

    gamma_S = 95.3
    gamma_D = 31

    # 32 states
    n = 32
    # State ordering: s1(-1,0,+1), s2(-2,-1,0,+1,+2), d0(0), d1(-1,0,+1),
    #                 d2(-2,-1,0,+1,+2), d3(-3,-2,-1,0,+1,+2,+3), p1(-1,0,+1), p2(-2,-1,0,+1,+2)

    states = [np.zeros(n) for _ in range(n)]
    for i in range(n):
        states[i][i] = 1

    # Name aliases for readability - state indices
    # s1: 0,1,2   s2: 3,4,5,6,7   d0: 8   d1: 9,10,11   d2: 12,13,14,15,16
    # d3: 17,18,19,20,21,22,23   p1: 24,25,26   p2: 27,28,29,30,31

    # Create density matrix index arrays
    cs = [np.kron(s, np.ones(n)) for s in states]
    cs_c = [np.kron(np.ones(n), s) for s in states]

    ind_array = np.eye(n)
    c_array = np.full((n, n * n), np.nan)
    c_conj_array = np.full((n, n * n), np.nan)
    for i in range(n):
        c_array[i, :] = np.kron(ind_array[i, :], np.ones(n))
        c_conj_array[i, :] = np.kron(np.ones(n), ind_array[i, :])

    # Shorthand for coefficients
    C = coeffs

    # Helper functions
    def _cp(idx):
        return cs[idx]

    def _cc(idx):
        return cs_c[idx]

    # Build derivative arrays for each state
    primes = [np.zeros(n * n, dtype=complex) for _ in range(n)]
    primes_c = [np.zeros(n * n, dtype=complex) for _ in range(n)]

    # --- S1 states (idx 0-2) ---
    # s1_1m (idx=0)
    primes[0] = (-1j * (Delta_S1P1 + Delta_S1P2 - Delta_zs1) * _cp(0)
                  - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m1m'] * pol_sp_0 * _cp(24) + C['A_S1P1_1m0'] * pol_sp_p * _cp(25))
                  - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m2m'] * pol_sp_m * _cp(27) + C['A_S1P2_1m1m'] * pol_sp_0 * _cp(28) + C['A_S1P2_1m0'] * pol_sp_p * _cp(29)))
    primes_c[0] = np.conj(-1j * (Delta_S1P1 + Delta_S1P2 - Delta_zs1) * _cc(0)
                           - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m1m'] * pol_sp_0 * _cc(24) + C['A_S1P1_1m0'] * pol_sp_p * _cc(25))
                           - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m2m'] * pol_sp_m * _cc(27) + C['A_S1P2_1m1m'] * pol_sp_0 * _cc(28) + C['A_S1P2_1m0'] * pol_sp_p * _cc(29)))

    # s1_0 (idx=1)
    primes[1] = (-1j * (Delta_S1P1 + Delta_S1P2) * _cp(1)
                  - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_01m'] * pol_sp_m * _cp(24) + C['A_S1P1_00'] * pol_sp_0 * _cp(25) + C['A_S1P1_01p'] * pol_sp_p * _cp(26))
                  - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_01m'] * pol_sp_m * _cp(28) + C['A_S1P2_00'] * pol_sp_0 * _cp(29) + C['A_S1P2_01p'] * pol_sp_p * _cp(30)))
    primes_c[1] = np.conj(-1j * (Delta_S1P1 + Delta_S1P2) * _cc(1)
                           - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_01m'] * pol_sp_m * _cc(24) + C['A_S1P1_00'] * pol_sp_0 * _cc(25) + C['A_S1P1_01p'] * pol_sp_p * _cc(26))
                           - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_01m'] * pol_sp_m * _cc(28) + C['A_S1P2_00'] * pol_sp_0 * _cc(29) + C['A_S1P2_01p'] * pol_sp_p * _cc(30)))

    # s1_1p (idx=2)
    primes[2] = (-1j * (Delta_S1P1 + Delta_S1P2 + Delta_zs1) * _cp(2)
                  - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1p0'] * pol_sp_m * _cp(25) + C['A_S1P1_1p1p'] * pol_sp_0 * _cp(26))
                  - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1p0'] * pol_sp_m * _cp(29) + C['A_S1P2_1p1p'] * pol_sp_0 * _cp(30) + C['A_S1P2_1p2p'] * pol_sp_p * _cp(31)))
    primes_c[2] = np.conj(-1j * (Delta_S1P1 + Delta_S1P2 + Delta_zs1) * _cc(2)
                           - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1p0'] * pol_sp_m * _cc(25) + C['A_S1P1_1p1p'] * pol_sp_0 * _cc(26))
                           - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1p0'] * pol_sp_m * _cc(29) + C['A_S1P2_1p1p'] * pol_sp_0 * _cc(30) + C['A_S1P2_1p2p'] * pol_sp_p * _cc(31)))

    # --- S2 states (idx 3-7) ---
    # s2_2m (idx=3)
    primes[3] = (-1j * (Delta_S2P1 + Delta_S2P2 - 2 * Delta_zs2) * _cp(3)
                  - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2m1m'] * pol_sp_p * _cp(24))
                  - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m2m'] * pol_sp_0 * _cp(27) + C['A_S2P2_2m1m'] * pol_sp_p * _cp(28)))
    primes_c[3] = np.conj(-1j * (Delta_S2P1 + Delta_S2P2 - 2 * Delta_zs2) * _cc(3)
                           - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2m1m'] * pol_sp_p * _cc(24))
                           - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m2m'] * pol_sp_0 * _cc(27) + C['A_S2P2_2m1m'] * pol_sp_p * _cc(28)))

    # s2_1m (idx=4)
    primes[4] = (-1j * (Delta_S2P1 + Delta_S2P2 - Delta_zs2) * _cp(4)
                  - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1m1m'] * pol_sp_0 * _cp(24) + C['A_S2P1_1m0'] * pol_sp_p * _cp(25))
                  - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1m2m'] * pol_sp_m * _cp(27) + C['A_S2P2_1m1m'] * pol_sp_0 * _cp(28) + C['A_S2P2_1m0'] * pol_sp_p * _cp(29)))
    primes_c[4] = np.conj(-1j * (Delta_S2P1 + Delta_S2P2 - Delta_zs2) * _cc(4)
                           - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1m1m'] * pol_sp_0 * _cc(24) + C['A_S2P1_1m0'] * pol_sp_p * _cc(25))
                           - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1m2m'] * pol_sp_m * _cc(27) + C['A_S2P2_1m1m'] * pol_sp_0 * _cc(28) + C['A_S2P2_1m0'] * pol_sp_p * _cc(29)))

    # s2_0 (idx=5)
    primes[5] = (-1j * (Delta_S2P1 + Delta_S2P2) * _cp(5)
                  - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_01m'] * pol_sp_m * _cp(24) + C['A_S2P1_00'] * pol_sp_0 * _cp(25) + C['A_S2P1_01p'] * pol_sp_p * _cp(26))
                  - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_01m'] * pol_sp_m * _cp(28) + C['A_S2P2_00'] * pol_sp_0 * _cp(29) + C['A_S2P2_01p'] * pol_sp_p * _cp(30)))
    primes_c[5] = np.conj(-1j * (Delta_S2P1 + Delta_S2P2) * _cc(5)
                           - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_01m'] * pol_sp_m * _cc(24) + C['A_S2P1_00'] * pol_sp_0 * _cc(25) + C['A_S2P1_01p'] * pol_sp_p * _cc(26))
                           - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_01m'] * pol_sp_m * _cc(28) + C['A_S2P2_00'] * pol_sp_0 * _cc(29) + C['A_S2P2_01p'] * pol_sp_p * _cc(30)))

    # s2_1p (idx=6)
    primes[6] = (-1j * (Delta_S2P1 + Delta_S2P2 + Delta_zs2) * _cp(6)
                  - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1p0'] * pol_sp_m * _cp(25) + C['A_S2P1_1p1p'] * pol_sp_0 * _cp(26))
                  - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1p0'] * pol_sp_m * _cp(29) + C['A_S2P2_1p1p'] * pol_sp_0 * _cp(30) + C['A_S2P2_1p2p'] * pol_sp_p * _cp(31)))
    primes_c[6] = np.conj(-1j * (Delta_S2P1 + Delta_S2P2 + Delta_zs2) * _cc(6)
                           - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1p0'] * pol_sp_m * _cc(25) + C['A_S2P1_1p1p'] * pol_sp_0 * _cc(26))
                           - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1p0'] * pol_sp_m * _cc(29) + C['A_S2P2_1p1p'] * pol_sp_0 * _cc(30) + C['A_S2P2_1p2p'] * pol_sp_p * _cc(31)))

    # s2_2p (idx=7)
    primes[7] = (-1j * (Delta_S2P1 + Delta_S2P2 + 2 * Delta_zs2) * _cp(7)
                  - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2p1p'] * pol_sp_m * _cp(26))
                  - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2p1p'] * pol_sp_m * _cp(30) + C['A_S2P2_2p2p'] * pol_sp_0 * _cp(31)))
    primes_c[7] = np.conj(-1j * (Delta_S2P1 + Delta_S2P2 + 2 * Delta_zs2) * _cc(7)
                           - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2p1p'] * pol_sp_m * _cc(26))
                           - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2p1p'] * pol_sp_m * _cc(30) + C['A_S2P2_2p2p'] * pol_sp_0 * _cc(31)))

    # --- D0 state (idx=8) ---
    primes[8] = (-1j * Delta_D0P1 * _cp(8)
                  - 1j * (Omega_D0P1 / 2) * (C['A_D0P1_01m'] * pol_dp_m * _cp(24) + C['A_D0P1_00'] * pol_dp_0 * _cp(25) + C['A_D0P1_01p'] * pol_dp_p * _cp(26)))
    primes_c[8] = np.conj(-1j * Delta_D0P1 * _cc(8)
                           - 1j * (Omega_D0P1 / 2) * (C['A_D0P1_01m'] * pol_dp_m * _cc(24) + C['A_D0P1_00'] * pol_dp_0 * _cc(25) + C['A_D0P1_01p'] * pol_dp_p * _cc(26)))

    # --- D1 states (idx 9-11) ---
    # d1_1m (idx=9)
    primes[9] = (-1j * (Delta_D1P1 + Delta_D1P2 - Delta_zd1) * _cp(9)
                  - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m1m'] * pol_dp_0 * _cp(24) + C['A_D1P1_1m0'] * pol_dp_p * _cp(25))
                  - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m2m'] * pol_dp_m * _cp(27) + C['A_D1P2_1m1m'] * pol_dp_0 * _cp(28) + C['A_D1P2_1m0'] * pol_dp_p * _cp(29)))
    primes_c[9] = np.conj(-1j * (Delta_D1P1 + Delta_D1P2 - Delta_zd1) * _cc(9)
                           - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m1m'] * pol_dp_0 * _cc(24) + C['A_D1P1_1m0'] * pol_dp_p * _cc(25))
                           - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m2m'] * pol_dp_m * _cc(27) + C['A_D1P2_1m1m'] * pol_dp_0 * _cc(28) + C['A_D1P2_1m0'] * pol_dp_p * _cc(29)))

    # d1_0 (idx=10)
    primes[10] = (-1j * (Delta_D1P1 + Delta_D1P2) * _cp(10)
                   - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_01m'] * pol_dp_m * _cp(24) + C['A_D1P1_00'] * pol_dp_0 * _cp(25) + C['A_D1P1_01p'] * pol_dp_p * _cp(26))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_01m'] * pol_dp_m * _cp(28) + C['A_D1P2_00'] * pol_dp_0 * _cp(29) + C['A_D1P2_01p'] * pol_dp_p * _cp(30)))
    primes_c[10] = np.conj(-1j * (Delta_D1P1 + Delta_D1P2) * _cc(10)
                            - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_01m'] * pol_dp_m * _cc(24) + C['A_D1P1_00'] * pol_dp_0 * _cc(25) + C['A_D1P1_01p'] * pol_dp_p * _cc(26))
                            - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_01m'] * pol_dp_m * _cc(28) + C['A_D1P2_00'] * pol_dp_0 * _cc(29) + C['A_D1P2_01p'] * pol_dp_p * _cc(30)))

    # d1_1p (idx=11)
    primes[11] = (-1j * (Delta_D1P1 + Delta_D1P2 + Delta_zd1) * _cp(11)
                   - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1p0'] * pol_dp_m * _cp(25) + C['A_D1P1_1p1p'] * pol_dp_0 * _cp(26))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1p0'] * pol_dp_m * _cp(29) + C['A_D1P2_1p1p'] * pol_dp_0 * _cp(30) + C['A_D1P2_1p2p'] * pol_dp_p * _cp(31)))
    primes_c[11] = np.conj(-1j * (Delta_D1P1 + Delta_D1P2 + Delta_zd1) * _cc(11)
                            - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1p0'] * pol_dp_m * _cc(25) + C['A_D1P1_1p1p'] * pol_dp_0 * _cc(26))
                            - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1p0'] * pol_dp_m * _cc(29) + C['A_D1P2_1p1p'] * pol_dp_0 * _cc(30) + C['A_D1P2_1p2p'] * pol_dp_p * _cc(31)))

    # --- D2 states (idx 12-16) ---
    # d2_2m (idx=12)
    primes[12] = (-1j * (Delta_D2P1 + Delta_D2P2 - 2 * Delta_zd2) * _cp(12)
                   - 1j * (Omega_D2P1 / 2) * C['A_D2P1_2m1m'] * pol_dp_p * _cp(24)
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m2m'] * pol_dp_0 * _cp(27) + C['A_D2P2_2m1m'] * pol_dp_p * _cp(28)))
    primes_c[12] = np.conj(-1j * (Delta_D2P1 + Delta_D2P2 - 2 * Delta_zd2) * _cc(12)
                            - 1j * (Omega_D2P1 / 2) * C['A_D2P1_2m1m'] * pol_dp_p * _cc(24)
                            - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m2m'] * pol_dp_0 * _cc(27) + C['A_D2P2_2m1m'] * pol_dp_p * _cc(28)))

    # d2_1m (idx=13)
    primes[13] = (-1j * (Delta_D2P1 + Delta_D2P2 - Delta_zd2) * _cp(13)
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1m1m'] * pol_dp_0 * _cp(24) + C['A_D2P1_1m0'] * pol_dp_p * _cp(25))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1m2m'] * pol_dp_m * _cp(27) + C['A_D2P2_1m1m'] * pol_dp_0 * _cp(28) + C['A_D2P2_1m0'] * pol_dp_p * _cp(29)))
    primes_c[13] = np.conj(-1j * (Delta_D2P1 + Delta_D2P2 - Delta_zd2) * _cc(13)
                            - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1m1m'] * pol_dp_0 * _cc(24) + C['A_D2P1_1m0'] * pol_dp_p * _cc(25))
                            - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1m2m'] * pol_dp_m * _cc(27) + C['A_D2P2_1m1m'] * pol_dp_0 * _cc(28) + C['A_D2P2_1m0'] * pol_dp_p * _cc(29)))

    # d2_0 (idx=14)
    primes[14] = (-1j * (Delta_D2P1 + Delta_D2P2) * _cp(14)
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_01m'] * pol_dp_m * _cp(24) + C['A_D2P1_00'] * pol_dp_0 * _cp(25) + C['A_D2P1_01p'] * pol_dp_p * _cp(26))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_01m'] * pol_dp_m * _cp(28) + C['A_D2P2_00'] * pol_dp_0 * _cp(29) + C['A_D2P2_01p'] * pol_dp_p * _cp(30)))
    primes_c[14] = np.conj(-1j * (Delta_D2P1 + Delta_D2P2) * _cc(14)
                            - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_01m'] * pol_dp_m * _cc(24) + C['A_D2P1_00'] * pol_dp_0 * _cc(25) + C['A_D2P1_01p'] * pol_dp_p * _cc(26))
                            - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_01m'] * pol_dp_m * _cc(28) + C['A_D2P2_00'] * pol_dp_0 * _cc(29) + C['A_D2P2_01p'] * pol_dp_p * _cc(30)))

    # d2_1p (idx=15)
    primes[15] = (-1j * (Delta_D2P1 + Delta_D2P2 + Delta_zd2) * _cp(15)
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1p0'] * pol_dp_m * _cp(25) + C['A_D2P1_1p1p'] * pol_dp_0 * _cp(26))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1p0'] * pol_dp_m * _cp(29) + C['A_D2P2_1p1p'] * pol_dp_0 * _cp(30) + C['A_D2P2_1p2p'] * pol_dp_p * _cp(31)))
    primes_c[15] = np.conj(-1j * (Delta_D2P1 + Delta_D2P2 + Delta_zd2) * _cc(15)
                            - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1p0'] * pol_dp_m * _cc(25) + C['A_D2P1_1p1p'] * pol_dp_0 * _cc(26))
                            - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1p0'] * pol_dp_m * _cc(29) + C['A_D2P2_1p1p'] * pol_dp_0 * _cc(30) + C['A_D2P2_1p2p'] * pol_dp_p * _cc(31)))

    # d2_2p (idx=16)
    primes[16] = (-1j * (Delta_D2P1 + Delta_D2P2 + 2 * Delta_zd2) * _cp(16)
                   - 1j * (Omega_D2P1 / 2) * C['A_D2P1_2p1p'] * pol_dp_m * _cp(26)
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2p1p'] * pol_dp_m * _cp(30) + C['A_D2P2_2p2p'] * pol_dp_0 * _cp(31)))
    primes_c[16] = np.conj(-1j * (Delta_D2P1 + Delta_D2P2 + 2 * Delta_zd2) * _cc(16)
                            - 1j * (Omega_D2P1 / 2) * C['A_D2P1_2p1p'] * pol_dp_m * _cc(26)
                            - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2p1p'] * pol_dp_m * _cc(30) + C['A_D2P2_2p2p'] * pol_dp_0 * _cc(31)))

    # --- D3 states (idx 17-23) ---
    # d3_3m (idx=17)
    primes[17] = (-1j * (Delta_D3P2 - 3 * Delta_zd3) * _cp(17)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3m2m'] * pol_dp_p * _cp(27)))
    primes_c[17] = np.conj(-1j * (Delta_D3P2 - 3 * Delta_zd3) * _cc(17)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3m2m'] * pol_dp_p * _cc(27)))

    # d3_2m (idx=18)
    primes[18] = (-1j * (Delta_D3P2 - 2 * Delta_zd3) * _cp(18)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2m2m'] * pol_dp_0 * _cp(27) + C['A_D3P2_2m1m'] * pol_dp_p * _cp(28)))
    primes_c[18] = np.conj(-1j * (Delta_D3P2 - 2 * Delta_zd3) * _cc(18)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2m2m'] * pol_dp_0 * _cc(27) + C['A_D3P2_2m1m'] * pol_dp_p * _cc(28)))

    # d3_1m (idx=19)
    primes[19] = (-1j * (Delta_D3P2 - Delta_zd3) * _cp(19)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1m2m'] * pol_dp_m * _cp(27) + C['A_D3P2_1m1m'] * pol_dp_0 * _cp(28) + C['A_D3P2_1m0'] * pol_dp_p * _cp(29)))
    primes_c[19] = np.conj(-1j * (Delta_D3P2 - Delta_zd3) * _cc(19)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1m2m'] * pol_dp_m * _cc(27) + C['A_D3P2_1m1m'] * pol_dp_0 * _cc(28) + C['A_D3P2_1m0'] * pol_dp_p * _cc(29)))

    # d3_0 (idx=20)
    primes[20] = (-1j * (Delta_D3P2) * _cp(20)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_01m'] * pol_dp_m * _cp(28) + C['A_D3P2_00'] * pol_dp_0 * _cp(29) + C['A_D3P2_01p'] * pol_dp_p * _cp(30)))
    primes_c[20] = np.conj(-1j * (Delta_D3P2) * _cc(20)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_01m'] * pol_dp_m * _cc(28) + C['A_D3P2_00'] * pol_dp_0 * _cc(29) + C['A_D3P2_01p'] * pol_dp_p * _cc(30)))

    # d3_1p (idx=21)
    primes[21] = (-1j * (Delta_D3P2 + Delta_zd3) * _cp(21)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1p0'] * pol_dp_m * _cp(29) + C['A_D3P2_1p1p'] * pol_dp_0 * _cp(30) + C['A_D3P2_1p2p'] * pol_dp_p * _cp(31)))
    primes_c[21] = np.conj(-1j * (Delta_D3P2 + Delta_zd3) * _cc(21)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1p0'] * pol_dp_m * _cc(29) + C['A_D3P2_1p1p'] * pol_dp_0 * _cc(30) + C['A_D3P2_1p2p'] * pol_dp_p * _cc(31)))

    # d3_2p (idx=22)
    primes[22] = (-1j * (Delta_D3P2 + 2 * Delta_zd3) * _cp(22)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2p1p'] * pol_dp_m * _cp(30) + C['A_D3P2_2p2p'] * pol_dp_0 * _cp(31)))
    primes_c[22] = np.conj(-1j * (Delta_D3P2 + 2 * Delta_zd3) * _cc(22)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2p1p'] * pol_dp_m * _cc(30) + C['A_D3P2_2p2p'] * pol_dp_0 * _cc(31)))

    # d3_3p (idx=23)
    primes[23] = (-1j * (Delta_D3P2 + 3 * Delta_zd3) * _cp(23)
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3p2p'] * pol_dp_m * _cp(31)))
    primes_c[23] = np.conj(-1j * (Delta_D3P2 + 3 * Delta_zd3) * _cc(23)
                            - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3p2p'] * pol_dp_m * _cc(31)))

    # --- P1 states (idx 24-26) ---
    # p1_1m (idx=24)
    primes[24] = (-1j * (-Delta_zp1) * _cp(24)
                   - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m1m'] * pol_sp_0 * _cp(0) + C['A_S1P1_01m'] * pol_sp_m * _cp(1))
                   - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2m1m'] * pol_sp_p * _cp(3) + C['A_S2P1_1m1m'] * pol_sp_0 * _cp(4) + C['A_S2P1_01m'] * pol_sp_m * _cp(5))
                   - 1j * (Omega_D0P1 / 2) * C['A_D0P1_01m'] * pol_dp_m * _cp(8)
                   - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m1m'] * pol_dp_0 * _cp(9) + C['A_D1P1_01m'] * pol_dp_m * _cp(10))
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_2m1m'] * pol_dp_p * _cp(12) + C['A_D2P1_1m1m'] * pol_dp_0 * _cp(13) + C['A_D2P1_01m'] * pol_dp_m * _cp(14)))
    primes_c[24] = np.conj(-1j * (-Delta_zp1) * _cc(24)
                             - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m1m'] * pol_sp_0 * _cc(0) + C['A_S1P1_01m'] * pol_sp_m * _cc(1))
                             - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_2m1m'] * pol_sp_p * _cc(3) + C['A_S2P1_1m1m'] * pol_sp_0 * _cc(4) + C['A_S2P1_01m'] * pol_sp_m * _cc(5))
                             - 1j * (Omega_D0P1 / 2) * C['A_D0P1_01m'] * pol_dp_m * _cc(8)
                             - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m1m'] * pol_dp_0 * _cc(9) + C['A_D1P1_01m'] * pol_dp_m * _cc(10))
                             - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_2m1m'] * pol_dp_p * _cc(12) + C['A_D2P1_1m1m'] * pol_dp_0 * _cc(13) + C['A_D2P1_01m'] * pol_dp_m * _cc(14)))

    # p1_0 (idx=25)
    primes[25] = (-1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m0'] * pol_sp_p * _cp(0) + C['A_S1P1_00'] * pol_sp_0 * _cp(1) + C['A_S1P1_1p0'] * pol_sp_m * _cp(2))
                   - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1m0'] * pol_sp_p * _cp(4) + C['A_S2P1_00'] * pol_sp_0 * _cp(5) + C['A_S2P1_1p0'] * pol_sp_m * _cp(6))
                   - 1j * (Omega_D0P1 / 2) * C['A_D0P1_00'] * pol_dp_0 * _cp(8)
                   - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m0'] * pol_dp_p * _cp(9) + C['A_D1P1_00'] * pol_dp_0 * _cp(10) + C['A_D1P1_1p0'] * pol_dp_m * _cp(11))
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1m0'] * pol_dp_p * _cp(13) + C['A_D2P1_00'] * pol_dp_0 * _cp(14) + C['A_D2P1_1p0'] * pol_dp_m * _cp(15)))
    primes_c[25] = np.conj(-1j * (Omega_S1P1 / 2) * (C['A_S1P1_1m0'] * pol_sp_p * _cc(0) + C['A_S1P1_00'] * pol_sp_0 * _cc(1) + C['A_S1P1_1p0'] * pol_sp_m * _cc(2))
                             - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_1m0'] * pol_sp_p * _cc(4) + C['A_S2P1_00'] * pol_sp_0 * _cc(5) + C['A_S2P1_1p0'] * pol_sp_m * _cc(6))
                             - 1j * (Omega_D0P1 / 2) * C['A_D0P1_00'] * pol_dp_0 * _cc(8)
                             - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_1m0'] * pol_dp_p * _cc(9) + C['A_D1P1_00'] * pol_dp_0 * _cc(10) + C['A_D1P1_1p0'] * pol_dp_m * _cc(11))
                             - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_1m0'] * pol_dp_p * _cc(13) + C['A_D2P1_00'] * pol_dp_0 * _cc(14) + C['A_D2P1_1p0'] * pol_dp_m * _cc(15)))

    # p1_1p (idx=26)
    primes[26] = (-1j * (Delta_zp1) * _cp(26)
                   - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_01p'] * pol_sp_p * _cp(1) + C['A_S1P1_1p1p'] * pol_sp_0 * _cp(2))
                   - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_01p'] * pol_sp_p * _cp(5) + C['A_S2P1_1p1p'] * pol_sp_0 * _cp(6) + C['A_S2P1_2p1p'] * pol_sp_m * _cp(7))
                   - 1j * (Omega_D0P1 / 2) * C['A_D0P1_01p'] * pol_dp_p * _cp(8)
                   - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_01p'] * pol_dp_p * _cp(10) + C['A_D1P1_1p1p'] * pol_dp_0 * _cp(11))
                   - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_01p'] * pol_dp_p * _cp(14) + C['A_D2P1_1p1p'] * pol_dp_0 * _cp(15) + C['A_D2P1_2p1p'] * pol_dp_m * _cp(16)))
    primes_c[26] = np.conj(-1j * (Delta_zp1) * _cc(26)
                             - 1j * (Omega_S1P1 / 2) * (C['A_S1P1_01p'] * pol_sp_p * _cc(1) + C['A_S1P1_1p1p'] * pol_sp_0 * _cc(2))
                             - 1j * (Omega_S2P1 / 2) * (C['A_S2P1_01p'] * pol_sp_p * _cc(5) + C['A_S2P1_1p1p'] * pol_sp_0 * _cc(6) + C['A_S2P1_2p1p'] * pol_sp_m * _cc(7))
                             - 1j * (Omega_D0P1 / 2) * C['A_D0P1_01p'] * pol_dp_p * _cc(8)
                             - 1j * (Omega_D1P1 / 2) * (C['A_D1P1_01p'] * pol_dp_p * _cc(10) + C['A_D1P1_1p1p'] * pol_dp_0 * _cc(11))
                             - 1j * (Omega_D2P1 / 2) * (C['A_D2P1_01p'] * pol_dp_p * _cc(14) + C['A_D2P1_1p1p'] * pol_dp_0 * _cc(15) + C['A_D2P1_2p1p'] * pol_dp_m * _cc(16)))

    # --- P2 states (idx 27-31) ---
    # p2_2m (idx=27)
    primes[27] = (-1j * (-2 * Delta_zp2) * _cp(27)
                   - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m2m'] * pol_sp_m * _cp(0))
                   - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m2m'] * pol_sp_0 * _cp(3) + C['A_S2P2_1m2m'] * pol_sp_m * _cp(4))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m2m'] * pol_dp_m * _cp(9))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m2m'] * pol_dp_0 * _cp(12) + C['A_D2P2_1m2m'] * pol_dp_m * _cp(13))
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3m2m'] * pol_dp_p * _cp(17) + C['A_D3P2_2m2m'] * pol_dp_0 * _cp(18) + C['A_D3P2_1m2m'] * pol_dp_m * _cp(19)))
    primes_c[27] = np.conj(-1j * (-2 * Delta_zp2) * _cc(27)
                             - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m2m'] * pol_sp_m * _cc(0))
                             - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m2m'] * pol_sp_0 * _cc(3) + C['A_S2P2_1m2m'] * pol_sp_m * _cc(4))
                             - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m2m'] * pol_dp_m * _cc(9))
                             - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m2m'] * pol_dp_0 * _cc(12) + C['A_D2P2_1m2m'] * pol_dp_m * _cc(13))
                             - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_3m2m'] * pol_dp_p * _cc(17) + C['A_D3P2_2m2m'] * pol_dp_0 * _cc(18) + C['A_D3P2_1m2m'] * pol_dp_m * _cc(19)))

    # p2_1m (idx=28)
    primes[28] = (-1j * (-Delta_zp2) * _cp(28)
                   - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m1m'] * pol_sp_0 * _cp(0) + C['A_S1P2_01m'] * pol_sp_m * _cp(1))
                   - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m1m'] * pol_sp_p * _cp(3) + C['A_S2P2_1m1m'] * pol_sp_0 * _cp(4) + C['A_S2P2_01m'] * pol_sp_m * _cp(5))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m1m'] * pol_dp_0 * _cp(9) + C['A_D1P2_01m'] * pol_dp_m * _cp(10))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m1m'] * pol_dp_p * _cp(12) + C['A_D2P2_1m1m'] * pol_dp_0 * _cp(13) + C['A_D2P2_01m'] * pol_dp_m * _cp(14))
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2m1m'] * pol_dp_p * _cp(18) + C['A_D3P2_1m1m'] * pol_dp_0 * _cp(19) + C['A_D3P2_01m'] * pol_dp_m * _cp(20)))
    primes_c[28] = np.conj(-1j * (-Delta_zp2) * _cc(28)
                             - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m1m'] * pol_sp_0 * _cc(0) + C['A_S1P2_01m'] * pol_sp_m * _cc(1))
                             - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_2m1m'] * pol_sp_p * _cc(3) + C['A_S2P2_1m1m'] * pol_sp_0 * _cc(4) + C['A_S2P2_01m'] * pol_sp_m * _cc(5))
                             - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m1m'] * pol_dp_0 * _cc(9) + C['A_D1P2_01m'] * pol_dp_m * _cc(10))
                             - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_2m1m'] * pol_dp_p * _cc(12) + C['A_D2P2_1m1m'] * pol_dp_0 * _cc(13) + C['A_D2P2_01m'] * pol_dp_m * _cc(14))
                             - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_2m1m'] * pol_dp_p * _cc(18) + C['A_D3P2_1m1m'] * pol_dp_0 * _cc(19) + C['A_D3P2_01m'] * pol_dp_m * _cc(20)))

    # p2_0 (idx=29)
    primes[29] = (-1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m0'] * pol_sp_p * _cp(0) + C['A_S1P2_00'] * pol_sp_0 * _cp(1) + C['A_S1P2_1p0'] * pol_sp_m * _cp(2))
                   - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1m0'] * pol_sp_p * _cp(4) + C['A_S2P2_00'] * pol_sp_0 * _cp(5) + C['A_S2P2_1p0'] * pol_sp_m * _cp(6))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m0'] * pol_dp_p * _cp(9) + C['A_D1P2_00'] * pol_dp_0 * _cp(10) + C['A_D1P2_1p0'] * pol_dp_m * _cp(11))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1m0'] * pol_dp_p * _cp(13) + C['A_D2P2_00'] * pol_dp_0 * _cp(14) + C['A_D2P2_1p0'] * pol_dp_m * _cp(15))
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1m0'] * pol_dp_p * _cp(19) + C['A_D3P2_00'] * pol_dp_0 * _cp(20) + C['A_D3P2_1p0'] * pol_dp_m * _cp(21)))
    primes_c[29] = np.conj(-1j * (Omega_S1P2 / 2) * (C['A_S1P2_1m0'] * pol_sp_p * _cc(0) + C['A_S1P2_00'] * pol_sp_0 * _cc(1) + C['A_S1P2_1p0'] * pol_sp_m * _cc(2))
                             - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1m0'] * pol_sp_p * _cc(4) + C['A_S2P2_00'] * pol_sp_0 * _cc(5) + C['A_S2P2_1p0'] * pol_sp_m * _cc(6))
                             - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1m0'] * pol_dp_p * _cc(9) + C['A_D1P2_00'] * pol_dp_0 * _cc(10) + C['A_D1P2_1p0'] * pol_dp_m * _cc(11))
                             - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1m0'] * pol_dp_p * _cc(13) + C['A_D2P2_00'] * pol_dp_0 * _cc(14) + C['A_D2P2_1p0'] * pol_dp_m * _cc(15))
                             - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1m0'] * pol_dp_p * _cc(19) + C['A_D3P2_00'] * pol_dp_0 * _cc(20) + C['A_D3P2_1p0'] * pol_dp_m * _cc(21)))

    # p2_1p (idx=30)
    primes[30] = (-1j * (Delta_zp2) * _cp(30)
                   - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_01p'] * pol_sp_p * _cp(1) + C['A_S1P2_1p1p'] * pol_sp_0 * _cp(2))
                   - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_01p'] * pol_sp_p * _cp(5) + C['A_S2P2_1p1p'] * pol_sp_0 * _cp(6) + C['A_S2P2_2p1p'] * pol_sp_m * _cp(7))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_01p'] * pol_dp_p * _cp(10) + C['A_D1P2_1p1p'] * pol_dp_0 * _cp(11))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_01p'] * pol_dp_p * _cp(14) + C['A_D2P2_1p1p'] * pol_dp_0 * _cp(15) + C['A_D2P2_2p1p'] * pol_dp_m * _cp(16))
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_01p'] * pol_dp_p * _cp(20) + C['A_D3P2_1p1p'] * pol_dp_0 * _cp(21) + C['A_D3P2_2p1p'] * pol_dp_m * _cp(22)))
    primes_c[30] = np.conj(-1j * (Delta_zp2) * _cc(30)
                             - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_01p'] * pol_sp_p * _cc(1) + C['A_S1P2_1p1p'] * pol_sp_0 * _cc(2))
                             - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_01p'] * pol_sp_p * _cc(5) + C['A_S2P2_1p1p'] * pol_sp_0 * _cc(6) + C['A_S2P2_2p1p'] * pol_sp_m * _cc(7))
                             - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_01p'] * pol_dp_p * _cc(10) + C['A_D1P2_1p1p'] * pol_dp_0 * _cc(11))
                             - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_01p'] * pol_dp_p * _cc(14) + C['A_D2P2_1p1p'] * pol_dp_0 * _cc(15) + C['A_D2P2_2p1p'] * pol_dp_m * _cc(16))
                             - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_01p'] * pol_dp_p * _cc(20) + C['A_D3P2_1p1p'] * pol_dp_0 * _cc(21) + C['A_D3P2_2p1p'] * pol_dp_m * _cc(22)))

    # p2_2p (idx=31)
    primes[31] = (-1j * (2 * Delta_zp2) * _cp(31)
                   - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1p2p'] * pol_sp_p * _cp(2))
                   - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1p2p'] * pol_sp_p * _cp(6) + C['A_S2P2_2p2p'] * pol_sp_0 * _cp(7))
                   - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1p2p'] * pol_dp_p * _cp(11))
                   - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1p2p'] * pol_dp_p * _cp(15) + C['A_D2P2_2p2p'] * pol_dp_0 * _cp(16))
                   - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1p2p'] * pol_dp_p * _cp(21) + C['A_D3P2_2p2p'] * pol_dp_0 * _cp(22) + C['A_D3P2_3p2p'] * pol_dp_m * _cp(23)))
    primes_c[31] = np.conj(-1j * (2 * Delta_zp2) * _cc(31)
                             - 1j * (Omega_S1P2 / 2) * (C['A_S1P2_1p2p'] * pol_sp_p * _cc(2))
                             - 1j * (Omega_S2P2 / 2) * (C['A_S2P2_1p2p'] * pol_sp_p * _cc(6) + C['A_S2P2_2p2p'] * pol_sp_0 * _cc(7))
                             - 1j * (Omega_D1P2 / 2) * (C['A_D1P2_1p2p'] * pol_dp_p * _cc(11))
                             - 1j * (Omega_D2P2 / 2) * (C['A_D2P2_1p2p'] * pol_dp_p * _cc(15) + C['A_D2P2_2p2p'] * pol_dp_0 * _cc(16))
                             - 1j * (Omega_D3P2 / 2) * (C['A_D3P2_1p2p'] * pol_dp_p * _cc(21) + C['A_D3P2_2p2p'] * pol_dp_0 * _cc(22) + C['A_D3P2_3p2p'] * pol_dp_m * _cc(23)))

    # --- Build evolution matrix ---
    ndm = n * n
    Evol_Matrix = np.full((ndm, ndm), np.nan, dtype=complex)

    # Generic off-diagonal elements with decoherence
    for i in range(n):
        for j in range(n):
            Pcount = 0
            if i >= 24:
                Pcount += 1
            if j >= 24:
                Pcount += 1
            mask = c_array[i, :] * c_conj_array[j, :] == 1
            Evol_Matrix[mask, :] = (c_array[i, :] * primes_c[j] + primes[i] * c_conj_array[j, :]
                                     - 0.5 * Pcount * (gamma_S + gamma_D) * c_array[i, :] * c_conj_array[j, :])

    # Override diagonal population equations with spontaneous emission terms
    W = C

    # --- S1 states (idx 0-2) ---
    # s1_1m (idx=0)
    mask = c_array[0, :] * c_conj_array[0, :] == 1
    Evol_Matrix[mask, :] = (c_array[0, :] * primes_c[0] + primes[0] * c_conj_array[0, :]
                             + W['W_S1P1_1m1m'] * gamma_S * c_array[24, :] * c_conj_array[24, :]
                             + W['W_S1P1_1m0'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S1P2_1m2m'] * gamma_S * c_array[27, :] * c_conj_array[27, :]
                             + W['W_S1P2_1m1m'] * gamma_S * c_array[28, :] * c_conj_array[28, :]
                             + W['W_S1P2_1m0'] * gamma_S * c_array[29, :] * c_conj_array[29, :])

    # s1_0 (idx=1)
    mask = c_array[1, :] * c_conj_array[1, :] == 1
    Evol_Matrix[mask, :] = (c_array[1, :] * primes_c[1] + primes[1] * c_conj_array[1, :]
                             + W['W_S1P1_01m'] * gamma_S * c_array[24, :] * c_conj_array[24, :]
                             + W['W_S1P1_00'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S1P1_01p'] * gamma_S * c_array[26, :] * c_conj_array[26, :]
                             + W['W_S1P2_01m'] * gamma_S * c_array[28, :] * c_conj_array[28, :]
                             + W['W_S1P2_00'] * gamma_S * c_array[29, :] * c_conj_array[29, :]
                             + W['W_S1P2_01p'] * gamma_S * c_array[30, :] * c_conj_array[30, :])

    # s1_1p (idx=2)
    mask = c_array[2, :] * c_conj_array[2, :] == 1
    Evol_Matrix[mask, :] = (c_array[2, :] * primes_c[2] + primes[2] * c_conj_array[2, :]
                             + W['W_S1P1_1p0'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S1P1_1p1p'] * gamma_S * c_array[26, :] * c_conj_array[26, :]
                             + W['W_S1P2_1p0'] * gamma_S * c_array[29, :] * c_conj_array[29, :]
                             + W['W_S1P2_1p1p'] * gamma_S * c_array[30, :] * c_conj_array[30, :]
                             + W['W_S1P2_1p2p'] * gamma_S * c_array[31, :] * c_conj_array[31, :])

    # --- S2 states (idx 3-7) ---
    # s2_2m (idx=3)
    mask = c_array[3, :] * c_conj_array[3, :] == 1
    Evol_Matrix[mask, :] = (c_array[3, :] * primes_c[3] + primes[3] * c_conj_array[3, :]
                             + W['W_S2P1_2m1m'] * gamma_S * c_array[24, :] * c_conj_array[24, :]
                             + W['W_S2P2_2m2m'] * gamma_S * c_array[27, :] * c_conj_array[27, :]
                             + W['W_S2P2_2m1m'] * gamma_S * c_array[28, :] * c_conj_array[28, :])

    # s2_1m (idx=4)
    mask = c_array[4, :] * c_conj_array[4, :] == 1
    Evol_Matrix[mask, :] = (c_array[4, :] * primes_c[4] + primes[4] * c_conj_array[4, :]
                             + W['W_S2P1_1m1m'] * gamma_S * c_array[24, :] * c_conj_array[24, :]
                             + W['W_S2P1_1m0'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S2P2_1m2m'] * gamma_S * c_array[27, :] * c_conj_array[27, :]
                             + W['W_S2P2_1m1m'] * gamma_S * c_array[28, :] * c_conj_array[28, :]
                             + W['W_S2P2_1m0'] * gamma_S * c_array[29, :] * c_conj_array[29, :])

    # s2_0 (idx=5)
    mask = c_array[5, :] * c_conj_array[5, :] == 1
    Evol_Matrix[mask, :] = (c_array[5, :] * primes_c[5] + primes[5] * c_conj_array[5, :]
                             + W['W_S2P1_01m'] * gamma_S * c_array[24, :] * c_conj_array[24, :]
                             + W['W_S2P1_00'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S2P1_01p'] * gamma_S * c_array[26, :] * c_conj_array[26, :]
                             + W['W_S2P2_01m'] * gamma_S * c_array[28, :] * c_conj_array[28, :]
                             + W['W_S2P2_00'] * gamma_S * c_array[29, :] * c_conj_array[29, :]
                             + W['W_S2P2_01p'] * gamma_S * c_array[30, :] * c_conj_array[30, :])

    # s2_1p (idx=6)
    mask = c_array[6, :] * c_conj_array[6, :] == 1
    Evol_Matrix[mask, :] = (c_array[6, :] * primes_c[6] + primes[6] * c_conj_array[6, :]
                             + W['W_S2P1_1p0'] * gamma_S * c_array[25, :] * c_conj_array[25, :]
                             + W['W_S2P1_1p1p'] * gamma_S * c_array[26, :] * c_conj_array[26, :]
                             + W['W_S2P2_1p0'] * gamma_S * c_array[29, :] * c_conj_array[29, :]
                             + W['W_S2P2_1p1p'] * gamma_S * c_array[30, :] * c_conj_array[30, :]
                             + W['W_S2P2_1p2p'] * gamma_S * c_array[31, :] * c_conj_array[31, :])

    # s2_2p (idx=7)
    mask = c_array[7, :] * c_conj_array[7, :] == 1
    Evol_Matrix[mask, :] = (c_array[7, :] * primes_c[7] + primes[7] * c_conj_array[7, :]
                             + W['W_S2P1_2p1p'] * gamma_S * c_array[26, :] * c_conj_array[26, :]
                             + W['W_S2P2_2p1p'] * gamma_S * c_array[30, :] * c_conj_array[30, :]
                             + W['W_S2P2_2p2p'] * gamma_S * c_array[31, :] * c_conj_array[31, :])

    # --- D0 state (idx=8) ---
    mask = c_array[8, :] * c_conj_array[8, :] == 1
    Evol_Matrix[mask, :] = (c_array[8, :] * primes_c[8] + primes[8] * c_conj_array[8, :]
                             + W['W_D0P1_01m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D0P1_00'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D0P1_01p'] * gamma_D * c_array[26, :] * c_conj_array[26, :])

    # --- D1 states (idx 9-11) ---
    # d1_1m (idx=9)
    mask = c_array[9, :] * c_conj_array[9, :] == 1
    Evol_Matrix[mask, :] = (c_array[9, :] * primes_c[9] + primes[9] * c_conj_array[9, :]
                             + W['W_D1P1_1m1m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D1P1_1m0'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D1P2_1m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :]
                             + W['W_D1P2_1m1m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D1P2_1m0'] * gamma_D * c_array[29, :] * c_conj_array[29, :])

    # d1_0 (idx=10)
    mask = c_array[10, :] * c_conj_array[10, :] == 1
    Evol_Matrix[mask, :] = (c_array[10, :] * primes_c[10] + primes[10] * c_conj_array[10, :]
                             + W['W_D1P1_01m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D1P1_00'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D1P1_01p'] * gamma_D * c_array[26, :] * c_conj_array[26, :]
                             + W['W_D1P2_01m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D1P2_00'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D1P2_01p'] * gamma_D * c_array[30, :] * c_conj_array[30, :])

    # d1_1p (idx=11)
    mask = c_array[11, :] * c_conj_array[11, :] == 1
    Evol_Matrix[mask, :] = (c_array[11, :] * primes_c[11] + primes[11] * c_conj_array[11, :]
                             + W['W_D1P1_1p0'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D1P1_1p1p'] * gamma_D * c_array[26, :] * c_conj_array[26, :]
                             + W['W_D1P2_1p0'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D1P2_1p1p'] * gamma_D * c_array[30, :] * c_conj_array[30, :]
                             + W['W_D1P2_1p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # --- D2 states (idx 12-16) ---
    # d2_2m (idx=12)
    mask = c_array[12, :] * c_conj_array[12, :] == 1
    Evol_Matrix[mask, :] = (c_array[12, :] * primes_c[12] + primes[12] * c_conj_array[12, :]
                             + W['W_D2P1_2m1m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D2P2_2m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :]
                             + W['W_D2P2_2m1m'] * gamma_D * c_array[28, :] * c_conj_array[28, :])

    # d2_1m (idx=13)
    mask = c_array[13, :] * c_conj_array[13, :] == 1
    Evol_Matrix[mask, :] = (c_array[13, :] * primes_c[13] + primes[13] * c_conj_array[13, :]
                             + W['W_D2P1_1m1m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D2P1_1m0'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D2P2_1m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :]
                             + W['W_D2P2_1m1m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D2P2_1m0'] * gamma_D * c_array[29, :] * c_conj_array[29, :])

    # d2_0 (idx=14)
    mask = c_array[14, :] * c_conj_array[14, :] == 1
    Evol_Matrix[mask, :] = (c_array[14, :] * primes_c[14] + primes[14] * c_conj_array[14, :]
                             + W['W_D2P1_01m'] * gamma_D * c_array[24, :] * c_conj_array[24, :]
                             + W['W_D2P1_00'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D2P1_01p'] * gamma_D * c_array[26, :] * c_conj_array[26, :]
                             + W['W_D2P2_01m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D2P2_00'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D2P2_01p'] * gamma_D * c_array[30, :] * c_conj_array[30, :])

    # d2_1p (idx=15)
    mask = c_array[15, :] * c_conj_array[15, :] == 1
    Evol_Matrix[mask, :] = (c_array[15, :] * primes_c[15] + primes[15] * c_conj_array[15, :]
                             + W['W_D2P1_1p0'] * gamma_D * c_array[25, :] * c_conj_array[25, :]
                             + W['W_D2P1_1p1p'] * gamma_D * c_array[26, :] * c_conj_array[26, :]
                             + W['W_D2P2_1p0'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D2P2_1p1p'] * gamma_D * c_array[30, :] * c_conj_array[30, :]
                             + W['W_D2P2_1p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # d2_2p (idx=16)
    mask = c_array[16, :] * c_conj_array[16, :] == 1
    Evol_Matrix[mask, :] = (c_array[16, :] * primes_c[16] + primes[16] * c_conj_array[16, :]
                             + W['W_D2P1_2p1p'] * gamma_D * c_array[26, :] * c_conj_array[26, :]
                             + W['W_D2P2_2p1p'] * gamma_D * c_array[30, :] * c_conj_array[30, :]
                             + W['W_D2P2_2p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # --- D3 states (idx 17-23) ---
    # d3_3m (idx=17)
    mask = c_array[17, :] * c_conj_array[17, :] == 1
    Evol_Matrix[mask, :] = (c_array[17, :] * primes_c[17] + primes[17] * c_conj_array[17, :]
                             + W['W_D3P2_3m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :])

    # d3_2m (idx=18)
    mask = c_array[18, :] * c_conj_array[18, :] == 1
    Evol_Matrix[mask, :] = (c_array[18, :] * primes_c[18] + primes[18] * c_conj_array[18, :]
                             + W['W_D3P2_2m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :]
                             + W['W_D3P2_2m1m'] * gamma_D * c_array[28, :] * c_conj_array[28, :])

    # d3_1m (idx=19)
    mask = c_array[19, :] * c_conj_array[19, :] == 1
    Evol_Matrix[mask, :] = (c_array[19, :] * primes_c[19] + primes[19] * c_conj_array[19, :]
                             + W['W_D3P2_1m2m'] * gamma_D * c_array[27, :] * c_conj_array[27, :]
                             + W['W_D3P2_1m1m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D3P2_1m0'] * gamma_D * c_array[29, :] * c_conj_array[29, :])

    # d3_0 (idx=20)
    mask = c_array[20, :] * c_conj_array[20, :] == 1
    Evol_Matrix[mask, :] = (c_array[20, :] * primes_c[20] + primes[20] * c_conj_array[20, :]
                             + W['W_D3P2_01m'] * gamma_D * c_array[28, :] * c_conj_array[28, :]
                             + W['W_D3P2_00'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D3P2_01p'] * gamma_D * c_array[30, :] * c_conj_array[30, :])

    # d3_1p (idx=21)
    mask = c_array[21, :] * c_conj_array[21, :] == 1
    Evol_Matrix[mask, :] = (c_array[21, :] * primes_c[21] + primes[21] * c_conj_array[21, :]
                             + W['W_D3P2_1p0'] * gamma_D * c_array[29, :] * c_conj_array[29, :]
                             + W['W_D3P2_1p1p'] * gamma_D * c_array[30, :] * c_conj_array[30, :]
                             + W['W_D3P2_1p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # d3_2p (idx=22)
    mask = c_array[22, :] * c_conj_array[22, :] == 1
    Evol_Matrix[mask, :] = (c_array[22, :] * primes_c[22] + primes[22] * c_conj_array[22, :]
                             + W['W_D3P2_2p1p'] * gamma_D * c_array[30, :] * c_conj_array[30, :]
                             + W['W_D3P2_2p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # d3_3p (idx=23)
    mask = c_array[23, :] * c_conj_array[23, :] == 1
    Evol_Matrix[mask, :] = (c_array[23, :] * primes_c[23] + primes[23] * c_conj_array[23, :]
                             + W['W_D3P2_3p2p'] * gamma_D * c_array[31, :] * c_conj_array[31, :])

    # Trace constraint
    trace_row = np.zeros((1, ndm), dtype=complex)
    for i in range(n):
        trace_row[0, c_array[i, :] * c_conj_array[i, :] == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    b = np.zeros(ndm + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end, c_array, c_conj_array, states[1]


if __name__ == "__main__":
    from ba137_spd_zeeman_levels_optical_bloch_init_prototype import ba137_init
    coeffs = ba137_init()
    sigma_end, c_array, c_conj_array, s1_0 = ba137_linsolve(
        coeffs, theta_s=0, theta_d=0,
        Omega_S1P2=2 * np.pi * 10, Omega_S2P2=2 * np.pi * 10,
        Omega_D0P1=2 * np.pi * 10, Omega_D1P1=2 * np.pi * 10,
        Omega_D2P2=2 * np.pi * 10, Omega_D3P2=2 * np.pi * 10,
        Delta_S1P2=-2 * np.pi * 20, Delta_D0P1=-2 * np.pi * 40)
    # Sum P state populations
    P_pop = sum(sigma_end[c_array[i, :] * c_conj_array[i, :] == 1].real for i in range(24, 32))
    print(f"Total P state population: {P_pop}")
