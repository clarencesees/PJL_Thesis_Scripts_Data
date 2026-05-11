import numpy as np
from math import sqrt, cos, sin


def _resolve_polarization(spec):
    """Return (pol_minus, pol_0, pol_plus) amplitudes for one laser.

    spec accepts either:
      - a scalar theta (rad): linear polarisation at angle theta from B,
        giving symmetric sigma+/sigma- amplitudes (cannot represent circular).
        Reproduces the original ``theta``-only API.
      - a length-3 sequence (pol_minus, pol_0, pol_plus): explicit amplitudes
        for the spherical components, allowing pure circular (e.g.
        (0, 0, 1) is sigma+ only) or any elliptical mixture. Amplitudes are
        used as-is; caller is responsible for normalisation
        (pol_minus^2 + pol_0^2 + pol_plus^2 = 1 for a unit-intensity field).
    """
    try:
        pm, p0, pp = spec
    except TypeError:
        theta = float(spec)
        c = sqrt(cos(theta) ** 2 / 2)
        return c, sqrt(sin(theta) ** 2), c
    return float(pm), float(p0), float(pp)


def ba138_spd_zeeman_linsolve_v2(Delta_SP01, Delta_DP01, Omega_S, Omega_D, theta_s, theta_d, B=835e-6, gamma_l=0.0):
    """
    Version 2: Separate polarization angles for S-P and D-P lasers.

    Parameters:
        Delta_SP01: S-P detuning (rad*MHz)
        Delta_DP01: D-P detuning (rad*MHz)
        Omega_S: S-P Rabi frequency (rad*MHz)
        Omega_D: D-P Rabi frequency (rad*MHz)
        theta_s: S-P laser polarisation. Either a scalar theta (rad) for linear
                 polarisation at angle theta from B, or a length-3 sequence of
                 amplitudes (pol_minus, pol_0, pol_plus) for arbitrary
                 polarisation including pure circular (e.g. (0, 0, 1) = sigma+).
        theta_d: D-P laser polarisation, same convention as theta_s.
        B: Magnetic field magnitude (Tesla). Default 835e-6 T (8.35 G).
        gamma_l: Laser-linewidth dephasing rate (rad*MHz, same units as gamma_S/gamma_D).
                 Per Dijck et al. PRA 91, 060501(R) (2015): adds dephasing to all
                 cross-manifold coherences (S-P and D-P at gamma + gamma_l;
                 S-D at gamma_l alone). Default 0 reproduces previous behavior.

    Returns:
        sigma_end: Steady-state density matrix elements
        cp0, cp0_c, cp1, cp1_c: P state index arrays
    """
    Planck_h = 6.62607015e-34
    uB = 9.274009994e-24
    hbar = Planck_h / (2 * np.pi)
    # Zeeman shifts in angular MHz (rad*MHz). Use hbar so units match Delta_SP01/Omega/gamma.
    Delta_zs = 1e-6 * 2 * uB * B / hbar
    Delta_zd = 1e-6 * (4 / 5) * uB * B / hbar
    Delta_zp = 1e-6 * (2 / 3) * uB * B / hbar

    pol_minus_s, pol_0_s, pol_plus_s = _resolve_polarization(theta_s)
    pol_minus_d, pol_0_d, pol_plus_d = _resolve_polarization(theta_d)

    gamma_S = 95.3
    gamma_D = 31

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
    cs0_prime = (-1j * Delta_SP01 * cs0
                 + 1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cp0 + sqrt(2 / 3) * pol_plus_s * cp1))
    cs0_prime_c = np.conj(-1j * Delta_SP01 * cs0_c
                           + 1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cp0_c + sqrt(2 / 3) * pol_plus_s * cp1_c))

    cs1_prime = (-1j * (Delta_SP01 + Delta_zs) * cs1
                 - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cp0 + sqrt(1 / 3) * pol_0_s * cp1))
    cs1_prime_c = np.conj(-1j * (Delta_SP01 + Delta_zs) * cs1_c
                           - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cp0_c + sqrt(1 / 3) * pol_0_s * cp1_c))

    cd0_prime = (-1j * Delta_DP01 * cd0
                 - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_plus_d * cp0)
    cd0_prime_c = np.conj(-1j * Delta_DP01 * cd0_c
                           - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_plus_d * cp0_c)

    cd1_prime = (-1j * (Delta_DP01 + Delta_zd) * cd1
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cp1 - sqrt(1 / 3) * pol_0_d * cp0))
    cd1_prime_c = np.conj(-1j * (Delta_DP01 + Delta_zd) * cd1_c
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cp1_c - sqrt(1 / 3) * pol_0_d * cp0_c))

    cd2_prime = (-1j * (Delta_DP01 + 2 * Delta_zd) * cd2
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_minus_d * cp0 - sqrt(1 / 3) * pol_0_d * cp1))
    cd2_prime_c = np.conj(-1j * (Delta_DP01 + 2 * Delta_zd) * cd2_c
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_minus_d * cp0_c - sqrt(1 / 3) * pol_0_d * cp1_c))

    cd3_prime = (-1j * (Delta_DP01 + 3 * Delta_zd) * cd3
                 - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_minus_d * cp1)
    cd3_prime_c = np.conj(-1j * (Delta_DP01 + 3 * Delta_zd) * cd3_c
                           - 1j * (Omega_D / 2) * sqrt(1 / 2) * pol_minus_d * cp1_c)

    cp0_prime = (1j * Delta_zp * cp0
                 - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cs1 - sqrt(1 / 3) * pol_0_s * cs0)
                 - 1j * (Omega_D / 2) * (sqrt(1 / 2) * pol_plus_d * cd0 - sqrt(1 / 3) * pol_0_d * cd1 + sqrt(1 / 6) * pol_minus_d * cd2))
    cp0_prime_c = np.conj(1j * Delta_zp * cp0_c
                           - 1j * (Omega_S / 2) * (sqrt(2 / 3) * pol_minus_s * cs1_c - sqrt(1 / 3) * pol_0_s * cs0_c)
                           - 1j * (Omega_D / 2) * (sqrt(1 / 2) * pol_plus_d * cd0_c - sqrt(1 / 3) * pol_0_d * cd1_c + sqrt(1 / 6) * pol_minus_d * cd2_c))

    cp1_prime = (-1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cs1 - sqrt(2 / 3) * pol_plus_s * cs0)
                 - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cd1 - sqrt(1 / 3) * pol_0_d * cd2 + sqrt(1 / 2) * pol_minus_d * cd3))
    cp1_prime_c = np.conj(-1j * (Omega_S / 2) * (sqrt(1 / 3) * pol_0_s * cs1_c - sqrt(2 / 3) * pol_plus_s * cs0_c)
                           - 1j * (Omega_D / 2) * (sqrt(1 / 6) * pol_plus_d * cd1_c - sqrt(1 / 3) * pol_0_d * cd2_c + sqrt(1 / 2) * pol_minus_d * cd3_c))

    primes = [cs0_prime, cs1_prime, cd0_prime, cd1_prime, cd2_prime, cd3_prime, cp0_prime, cp1_prime]
    primes_c = [cs0_prime_c, cs1_prime_c, cd0_prime_c, cd1_prime_c, cd2_prime_c, cd3_prime_c, cp0_prime_c, cp1_prime_c]

    ndm = n * n
    Evol_Matrix = np.full((ndm, ndm), np.nan, dtype=complex)

    # Diagonal populations with decay
    Evol_Matrix[cs0 * cs0_c == 1, :] = (cs0 * cs0_prime_c + cs0_prime * cs0_c
                                          + (1 / 3) * gamma_S * cp0 * cp0_c + (2 / 3) * gamma_S * cp1 * cp1_c)
    Evol_Matrix[cs1 * cs1_c == 1, :] = (cs1 * cs1_prime_c + cs1_prime * cs1_c
                                          + (2 / 3) * gamma_S * cp0 * cp0_c + (1 / 3) * gamma_S * cp1 * cp1_c)
    Evol_Matrix[cd0 * cd0_c == 1, :] = cd0 * cd0_prime_c + cd0_prime * cd0_c + (1 / 2) * gamma_D * cp0 * cp0_c
    Evol_Matrix[cd1 * cd1_c == 1, :] = (cd1 * cd1_prime_c + cd1_prime * cd1_c
                                          + (1 / 3) * gamma_D * cp0 * cp0_c + (1 / 6) * gamma_D * cp1 * cp1_c)
    Evol_Matrix[cd2 * cd2_c == 1, :] = (cd2 * cd2_prime_c + cd2_prime * cd2_c
                                          + (1 / 6) * gamma_D * cp0 * cp0_c + (1 / 3) * gamma_D * cp1 * cp1_c)
    Evol_Matrix[cd3 * cd3_c == 1, :] = cd3 * cd3_prime_c + cd3_prime * cd3_c + (1 / 2) * gamma_D * cp1 * cp1_c
    Evol_Matrix[cp0 * cp0_c == 1, :] = cp0 * cp0_prime_c + cp0_prime * cp0_c - (gamma_S + gamma_D) * cp0 * cp0_c
    Evol_Matrix[cp1 * cp1_c == 1, :] = cp1 * cp1_prime_c + cp1_prime * cp1_c - (gamma_S + gamma_D) * cp1 * cp1_c

    # Manifold tags for cross-manifold (gamma_l) dephasing.
    # 0 = S (cs0, cs1), 1 = D (cd0..cd3), 2 = P (cp0, cp1)
    manifold = (0, 0, 1, 1, 1, 1, 2, 2)

    # Off-diagonal coherences. Per Dijck 2015 R-matrix:
    #   one P involved (S-P, D-P): damping = gamma + gamma_l
    #   two P involved (P-P):      damping = 2 gamma  (no gamma_l, both addressed by same laser pair)
    #   no P involved, cross-manifold (S-D): damping = gamma_l only
    #   same-manifold off-diagonals (S-S, D-D): no extra damping (only feed terms via primes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mask = c[i] * c_conj[j] == 1
            p_count = (1 if i >= 6 else 0) + (1 if j >= 6 else 0)
            cross = (manifold[i] != manifold[j])
            gamma_l_contrib = gamma_l if (cross and p_count < 2) else 0.0
            damp = (p_count / 2) * (gamma_S + gamma_D) + gamma_l_contrib
            Evol_Matrix[mask, :] = (c[i] * primes_c[j] + primes[i] * c_conj[j]
                                     - damp * c[i] * c_conj[j])

    # Trace constraint
    trace_row = np.zeros((1, ndm), dtype=complex)
    for i in range(n):
        trace_row[0, c[i] * c_conj[i] == 1] = 1
    Evol_Matrix = np.vstack([Evol_Matrix, trace_row])

    b = np.zeros(ndm + 1, dtype=complex)
    b[-1] = 1
    sigma_end, _, _, _ = np.linalg.lstsq(Evol_Matrix, b, rcond=None)

    return sigma_end, cp0, cp0_c, cp1, cp1_c
