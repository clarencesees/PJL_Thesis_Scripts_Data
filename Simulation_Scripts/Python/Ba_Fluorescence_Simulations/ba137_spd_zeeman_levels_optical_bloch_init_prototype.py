"""
Initializes the Clebsch-Gordan and Wigner-Eckart coefficients required for
Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.

Naming convention:
A_XXYY_xxyy: Clebsch-Gordan coefficient for transition from XX level (F,m) to YY level (F,m)
W_XXYY_xxyy: Wigner-Eckart coefficient for the same transition
  S[X] = S_{1/2}, F=[X]; P[X] = P_{1/2}, F=[X]; D[X] = D_{3/2}, F=[X]
  m/p denote minus/plus for Zeeman levels; 0 = m=0

Requires: get_clebsch_gordan.py, fm_to_j_coefficient.py, not_wigner_6j.py
"""
import numpy as np
from fractions import Fraction
from get_clebsch_gordan import get_clebsch_gordan
from fm_to_j_coefficient import fm_to_j_coefficient


def _rat_correct(val):
    """Correct round-off error using rational approximation (like MATLAB's rat)."""
    if abs(val) < 1e-10:
        return 0.0
    sign = 1 if val >= 0 else -1
    frac = Fraction(val**2).limit_denominator(1000)
    return sign * np.sqrt(float(frac))


def ba137_init():
    """
    Initialize all Clebsch-Gordan and Wigner-Eckart coefficients for Ba-137.

    Returns:
        coeffs: dict containing all A_ and W_ coefficients, plus initialized
                Omega and Delta values.
    """
    coeffs = {}

    # --- CG coefficients for J1=0, J2=1 (D0->P1 transitions) ---
    A, J_index, Jm_index, M1_index, M2_index = get_clebsch_gordan(0, 1)
    A[np.abs(A) < 1e-10] = 0
    for i in range(A.size):
        A.flat[i] = _rat_correct(A.flat[i])

    coeffs['A_D0P1_01m'] = A[(np.abs(M1_index - 0) < 1e-10) & (np.abs(M2_index - (-1)) < 1e-10)][:, (np.abs(J_index - 1) < 1e-10) & (np.abs(Jm_index - (-1)) < 1e-10)].item()
    coeffs['A_D0P1_00'] = A[(np.abs(M1_index - 0) < 1e-10) & (np.abs(M2_index - 0) < 1e-10)][:, (np.abs(J_index - 1) < 1e-10) & (np.abs(Jm_index - 0) < 1e-10)].item()
    coeffs['A_D0P1_01p'] = A[(np.abs(M1_index - 0) < 1e-10) & (np.abs(M2_index - 1) < 1e-10)][:, (np.abs(J_index - 1) < 1e-10) & (np.abs(Jm_index - 1) < 1e-10)].item()

    # --- CG coefficients for J1=1, J2=1 (F=1 transitions) ---
    A, J_index, Jm_index, M1_index, M2_index = get_clebsch_gordan(1, 1)
    A[np.abs(A) < 1e-10] = 0
    for i in range(A.size):
        A.flat[i] = _rat_correct(A.flat[i])

    def _cg(m1, m2, j, jm):
        row = (np.abs(M1_index - m1) < 1e-10) & (np.abs(M2_index - m2) < 1e-10)
        col = (np.abs(J_index - j) < 1e-10) & (np.abs(Jm_index - jm) < 1e-10)
        val = A[row][:, col]
        return val.item() if val.size > 0 else 0.0

    # S1->P1 and D1->P1
    for prefix in ['S1P1', 'D1P1']:
        coeffs[f'A_{prefix}_1m1m'] = _cg(-1, 0, 1, -1)
        coeffs[f'A_{prefix}_1m0'] = _cg(-1, 1, 1, 0)
        coeffs[f'A_{prefix}_01m'] = _cg(0, -1, 1, -1)
        coeffs[f'A_{prefix}_00'] = _cg(0, 0, 1, 0)
        coeffs[f'A_{prefix}_01p'] = _cg(0, 1, 1, 1)
        coeffs[f'A_{prefix}_1p0'] = _cg(1, -1, 1, 0)
        coeffs[f'A_{prefix}_1p1p'] = _cg(1, 0, 1, 1)

    # S1->P2 and D1->P2
    for prefix in ['S1P2', 'D1P2']:
        coeffs[f'A_{prefix}_1m2m'] = _cg(-1, -1, 2, -2)
        coeffs[f'A_{prefix}_1m1m'] = _cg(-1, 0, 2, -1)
        coeffs[f'A_{prefix}_1m0'] = _cg(-1, 1, 2, 0)
        coeffs[f'A_{prefix}_01m'] = _cg(0, -1, 2, -1)
        coeffs[f'A_{prefix}_00'] = _cg(0, 0, 2, 0)
        coeffs[f'A_{prefix}_01p'] = _cg(0, 1, 2, 1)
        coeffs[f'A_{prefix}_1p0'] = _cg(1, -1, 2, 0)
        coeffs[f'A_{prefix}_1p1p'] = _cg(1, 0, 2, 1)
        coeffs[f'A_{prefix}_1p2p'] = _cg(1, 1, 2, 2)

    # --- CG coefficients for J1=2, J2=1 (F=2 transitions) ---
    A, J_index, Jm_index, M1_index, M2_index = get_clebsch_gordan(2, 1)
    A[np.abs(A) < 1e-10] = 0
    for i in range(A.size):
        A.flat[i] = _rat_correct(A.flat[i])

    def _cg2(m1, m2, j, jm):
        row = (np.abs(M1_index - m1) < 1e-10) & (np.abs(M2_index - m2) < 1e-10)
        col = (np.abs(J_index - j) < 1e-10) & (np.abs(Jm_index - jm) < 1e-10)
        val = A[row][:, col]
        return val.item() if val.size > 0 else 0.0

    for prefix in ['S2P1', 'D2P1']:
        coeffs[f'A_{prefix}_2m1m'] = _cg2(-2, 1, 1, -1)
        coeffs[f'A_{prefix}_1m1m'] = _cg2(-1, 0, 1, -1)
        coeffs[f'A_{prefix}_1m0'] = _cg2(-1, 1, 1, 0)
        coeffs[f'A_{prefix}_01m'] = _cg2(0, -1, 1, -1)
        coeffs[f'A_{prefix}_00'] = _cg2(0, 0, 1, 0)
        coeffs[f'A_{prefix}_01p'] = _cg2(0, 1, 1, 1)
        coeffs[f'A_{prefix}_1p0'] = _cg2(1, -1, 1, 0)
        coeffs[f'A_{prefix}_1p1p'] = _cg2(1, 0, 1, 1)
        coeffs[f'A_{prefix}_2p1p'] = _cg2(2, -1, 1, 1)

    for prefix in ['S2P2', 'D2P2']:
        coeffs[f'A_{prefix}_2m2m'] = _cg2(-2, 0, 2, -2)
        coeffs[f'A_{prefix}_2m1m'] = _cg2(-2, 1, 2, -1)
        coeffs[f'A_{prefix}_1m2m'] = _cg2(-1, -1, 2, -2)
        coeffs[f'A_{prefix}_1m1m'] = _cg2(-1, 0, 2, -1)
        coeffs[f'A_{prefix}_1m0'] = _cg2(-1, 1, 2, 0)
        coeffs[f'A_{prefix}_01m'] = _cg2(0, -1, 2, -1)
        coeffs[f'A_{prefix}_00'] = _cg2(0, 0, 2, 0)
        coeffs[f'A_{prefix}_01p'] = _cg2(0, 1, 2, 1)
        coeffs[f'A_{prefix}_1p0'] = _cg2(1, -1, 2, 0)
        coeffs[f'A_{prefix}_1p1p'] = _cg2(1, 0, 2, 1)
        coeffs[f'A_{prefix}_1p2p'] = _cg2(1, 1, 2, 2)
        coeffs[f'A_{prefix}_2p1p'] = _cg2(2, -1, 2, 1)
        coeffs[f'A_{prefix}_2p2p'] = _cg2(2, 0, 2, 2)

    # --- CG coefficients for J1=3, J2=1 (D3->P2 transitions) ---
    A, J_index, Jm_index, M1_index, M2_index = get_clebsch_gordan(3, 1)
    A[np.abs(A) < 1e-10] = 0
    for i in range(A.size):
        A.flat[i] = _rat_correct(A.flat[i])

    def _cg3(m1, m2, j, jm):
        row = (np.abs(M1_index - m1) < 1e-10) & (np.abs(M2_index - m2) < 1e-10)
        col = (np.abs(J_index - j) < 1e-10) & (np.abs(Jm_index - jm) < 1e-10)
        val = A[row][:, col]
        return val.item() if val.size > 0 else 0.0

    coeffs['A_D3P2_3m2m'] = _cg3(-3, 1, 2, -2)
    coeffs['A_D3P2_2m2m'] = _cg3(-2, 0, 2, -2)
    coeffs['A_D3P2_2m1m'] = _cg3(-2, 1, 2, -1)
    coeffs['A_D3P2_1m2m'] = _cg3(-1, -1, 2, -2)
    coeffs['A_D3P2_1m1m'] = _cg3(-1, 0, 2, -1)
    coeffs['A_D3P2_1m0'] = _cg3(-1, 1, 2, 0)
    coeffs['A_D3P2_01m'] = _cg3(0, -1, 2, -1)
    coeffs['A_D3P2_00'] = _cg3(0, 0, 2, 0)
    coeffs['A_D3P2_01p'] = _cg3(0, 1, 2, 1)
    coeffs['A_D3P2_1p0'] = _cg3(1, -1, 2, 0)
    coeffs['A_D3P2_1p1p'] = _cg3(1, 0, 2, 1)
    coeffs['A_D3P2_1p2p'] = _cg3(1, 1, 2, 2)
    coeffs['A_D3P2_2p1p'] = _cg3(2, -1, 2, 1)
    coeffs['A_D3P2_2p2p'] = _cg3(2, 0, 2, 2)
    coeffs['A_D3P2_3p2p'] = _cg3(3, -1, 2, 2)

    # --- Wigner-Eckart coefficients ---
    S1, S2, D0, D1, D2, D3, P1, P2 = 1, 2, 0, 1, 2, 3, 1, 2
    m3, m2, m1, p0, p1, p2, p3 = -3, -2, -1, 0, 1, 2, 3
    S_J, D_J, P_J = 0.5, 1.5, 0.5
    I_spin = 1.5

    def _w(F1, mF1, F2, mF2, J1, J2):
        val = fm_to_j_coefficient(F1, mF1, F2, mF2, 1, mF1 - mF2, P_J if 'P' in '' else J1, J2, I_spin)
        return float(Fraction(val**2).limit_denominator(1000))

    # Compute W coefficients using fm_to_j_coefficient
    def _w_coeff(F1_val, mF1_val, F2_val, mF2_val, J1_val, J2_val):
        val = fm_to_j_coefficient(F1_val, mF1_val, F2_val, mF2_val, 1, mF1_val - mF2_val, J1_val, J2_val, I_spin)
        result = val ** 2
        frac = Fraction(float(result)).limit_denominator(1000)
        return float(frac)

    # W_S1P1 coefficients
    coeffs['W_S1P1_1m1m'] = _w_coeff(P1, m1, S1, m1, P_J, S_J)
    coeffs['W_S1P1_1m0'] = _w_coeff(P1, p0, S1, m1, P_J, S_J)
    coeffs['W_S1P1_01m'] = _w_coeff(P1, m1, S1, p0, P_J, S_J)
    coeffs['W_S1P1_00'] = _w_coeff(P1, p0, S1, p0, P_J, S_J)
    coeffs['W_S1P1_01p'] = _w_coeff(P1, p1, S1, p0, P_J, S_J)
    coeffs['W_S1P1_1p0'] = _w_coeff(P1, p0, S1, p1, P_J, S_J)
    coeffs['W_S1P1_1p1p'] = _w_coeff(P1, p1, S1, p1, P_J, S_J)

    # W_S1P2 coefficients
    coeffs['W_S1P2_1m2m'] = _w_coeff(P2, m2, S1, m1, P_J, S_J)
    coeffs['W_S1P2_1m1m'] = _w_coeff(P2, m1, S1, m1, P_J, S_J)
    coeffs['W_S1P2_1m0'] = _w_coeff(P2, p0, S1, m1, P_J, S_J)
    coeffs['W_S1P2_01m'] = _w_coeff(P2, m1, S1, p0, P_J, S_J)
    coeffs['W_S1P2_00'] = _w_coeff(P2, p0, S1, p0, P_J, S_J)
    coeffs['W_S1P2_01p'] = _w_coeff(P2, p1, S1, p0, P_J, S_J)
    coeffs['W_S1P2_1p0'] = _w_coeff(P2, p0, S1, p1, P_J, S_J)
    coeffs['W_S1P2_1p1p'] = _w_coeff(P2, p1, S1, p1, P_J, S_J)
    coeffs['W_S1P2_1p2p'] = _w_coeff(P2, p2, S1, p1, P_J, S_J)

    # W_S2P1 coefficients
    coeffs['W_S2P1_2m1m'] = _w_coeff(P1, m1, S2, m2, P_J, S_J)
    coeffs['W_S2P1_1m1m'] = _w_coeff(P1, m1, S2, m1, P_J, S_J)
    coeffs['W_S2P1_1m0'] = _w_coeff(P1, p0, S2, m1, P_J, S_J)
    coeffs['W_S2P1_01m'] = _w_coeff(P1, m1, S2, p0, P_J, S_J)
    coeffs['W_S2P1_00'] = _w_coeff(P1, p0, S2, p0, P_J, S_J)
    coeffs['W_S2P1_01p'] = _w_coeff(P1, p1, S2, p0, P_J, S_J)
    coeffs['W_S2P1_1p0'] = _w_coeff(P1, p0, S2, p1, P_J, S_J)
    coeffs['W_S2P1_1p1p'] = _w_coeff(P1, p1, S2, p1, P_J, S_J)
    coeffs['W_S2P1_2p1p'] = _w_coeff(P1, p1, S2, p2, P_J, S_J)

    # W_S2P2 coefficients
    coeffs['W_S2P2_2m2m'] = _w_coeff(P2, m2, S2, m2, P_J, S_J)
    coeffs['W_S2P2_2m1m'] = _w_coeff(P2, m1, S2, m2, P_J, S_J)
    coeffs['W_S2P2_1m2m'] = _w_coeff(P2, m2, S2, m1, P_J, S_J)
    coeffs['W_S2P2_1m1m'] = _w_coeff(P2, m1, S2, m1, P_J, S_J)
    coeffs['W_S2P2_1m0'] = _w_coeff(P2, p0, S2, m1, P_J, S_J)
    coeffs['W_S2P2_01m'] = _w_coeff(P2, m1, S2, p0, P_J, S_J)
    coeffs['W_S2P2_00'] = _w_coeff(P2, p0, S2, p0, P_J, S_J)
    coeffs['W_S2P2_01p'] = _w_coeff(P2, p1, S2, p0, P_J, S_J)
    coeffs['W_S2P2_1p0'] = _w_coeff(P2, p0, S2, p1, P_J, S_J)
    coeffs['W_S2P2_1p1p'] = _w_coeff(P2, p1, S2, p1, P_J, S_J)
    coeffs['W_S2P2_1p2p'] = _w_coeff(P2, p2, S2, p1, P_J, S_J)
    coeffs['W_S2P2_2p1p'] = _w_coeff(P2, p1, S2, p2, P_J, S_J)
    coeffs['W_S2P2_2p2p'] = _w_coeff(P2, p2, S2, p2, P_J, S_J)

    # W_D0P1 coefficients
    coeffs['W_D0P1_01m'] = _w_coeff(P1, m1, D0, p0, P_J, D_J)
    coeffs['W_D0P1_00'] = _w_coeff(P1, p0, D0, p0, P_J, D_J)
    coeffs['W_D0P1_01p'] = _w_coeff(P1, p1, D0, p0, P_J, D_J)

    # W_D1P1 coefficients
    coeffs['W_D1P1_1m1m'] = _w_coeff(P1, m1, D1, p0, P_J, D_J)
    coeffs['W_D1P1_1m0'] = _w_coeff(P1, p0, D1, m1, P_J, D_J)
    coeffs['W_D1P1_01m'] = _w_coeff(P1, m1, D1, p0, P_J, D_J)
    coeffs['W_D1P1_00'] = _w_coeff(P1, p0, D1, p0, P_J, D_J)
    coeffs['W_D1P1_01p'] = _w_coeff(P1, p1, D1, p0, P_J, D_J)
    coeffs['W_D1P1_1p0'] = _w_coeff(P1, p0, D1, p1, P_J, D_J)
    coeffs['W_D1P1_1p1p'] = _w_coeff(P1, p1, D1, p1, P_J, D_J)

    # W_D1P2 coefficients
    coeffs['W_D1P2_1m2m'] = _w_coeff(P2, m2, D1, m1, P_J, D_J)
    coeffs['W_D1P2_1m1m'] = _w_coeff(P2, m1, D1, m1, P_J, D_J)
    coeffs['W_D1P2_1m0'] = _w_coeff(P2, p0, D1, m1, P_J, D_J)
    coeffs['W_D1P2_01m'] = _w_coeff(P2, m1, D1, p0, P_J, D_J)
    coeffs['W_D1P2_00'] = _w_coeff(P2, p0, D1, p0, P_J, D_J)
    coeffs['W_D1P2_01p'] = _w_coeff(P2, p1, D1, p0, P_J, D_J)
    coeffs['W_D1P2_1p0'] = _w_coeff(P2, p0, D1, p1, P_J, D_J)
    coeffs['W_D1P2_1p1p'] = _w_coeff(P2, p1, D1, p1, P_J, D_J)
    coeffs['W_D1P2_1p2p'] = _w_coeff(P2, p2, D1, p1, P_J, D_J)

    # W_D2P1 coefficients
    coeffs['W_D2P1_2m1m'] = _w_coeff(P1, m1, D2, m2, P_J, D_J)
    coeffs['W_D2P1_1m1m'] = _w_coeff(P1, m1, D2, m1, P_J, D_J)
    coeffs['W_D2P1_1m0'] = _w_coeff(P1, p0, D2, m1, P_J, D_J)
    coeffs['W_D2P1_01m'] = _w_coeff(P1, m1, D2, p0, P_J, D_J)
    coeffs['W_D2P1_00'] = _w_coeff(P1, p0, D2, p0, P_J, D_J)
    coeffs['W_D2P1_01p'] = _w_coeff(P1, p1, D2, p0, P_J, D_J)
    coeffs['W_D2P1_1p0'] = _w_coeff(P1, p0, D2, p1, P_J, D_J)
    coeffs['W_D2P1_1p1p'] = _w_coeff(P1, p1, D2, p1, P_J, D_J)
    coeffs['W_D2P1_2p1p'] = _w_coeff(P1, p1, D2, p2, P_J, D_J)

    # W_D2P2 coefficients
    coeffs['W_D2P2_2m2m'] = _w_coeff(P2, m2, D2, m2, P_J, D_J)
    coeffs['W_D2P2_2m1m'] = _w_coeff(P2, m1, D2, m2, P_J, D_J)
    coeffs['W_D2P2_1m2m'] = _w_coeff(P2, m2, D2, m1, P_J, D_J)
    coeffs['W_D2P2_1m1m'] = _w_coeff(P2, m1, D2, m1, P_J, D_J)
    coeffs['W_D2P2_1m0'] = _w_coeff(P2, p0, D2, m1, P_J, D_J)
    coeffs['W_D2P2_01m'] = _w_coeff(P2, m1, D2, p0, P_J, D_J)
    coeffs['W_D2P2_00'] = _w_coeff(P2, p0, D2, p0, P_J, D_J)
    coeffs['W_D2P2_01p'] = _w_coeff(P2, p1, D2, p0, P_J, D_J)
    coeffs['W_D2P2_1p0'] = _w_coeff(P2, p0, D2, p1, P_J, D_J)
    coeffs['W_D2P2_1p1p'] = _w_coeff(P2, p1, D2, p1, P_J, D_J)
    coeffs['W_D2P2_1p2p'] = _w_coeff(P2, p2, D2, p1, P_J, D_J)
    coeffs['W_D2P2_2p1p'] = _w_coeff(P2, p1, D2, p2, P_J, D_J)
    coeffs['W_D2P2_2p2p'] = _w_coeff(P2, p2, D2, p2, P_J, D_J)

    # W_D3P2 coefficients
    coeffs['W_D3P2_3m2m'] = _w_coeff(P2, m2, D3, m3, P_J, D_J)
    coeffs['W_D3P2_2m2m'] = _w_coeff(P2, m2, D3, m2, P_J, D_J)
    coeffs['W_D3P2_2m1m'] = _w_coeff(P2, m1, D3, m2, P_J, D_J)
    coeffs['W_D3P2_1m2m'] = _w_coeff(P2, m2, D3, m1, P_J, D_J)
    coeffs['W_D3P2_1m1m'] = _w_coeff(P2, m1, D3, m1, P_J, D_J)
    coeffs['W_D3P2_1m0'] = _w_coeff(P2, p0, D3, m1, P_J, D_J)
    coeffs['W_D3P2_01m'] = _w_coeff(P2, m1, D3, p0, P_J, D_J)
    coeffs['W_D3P2_00'] = _w_coeff(P2, p0, D3, p0, P_J, D_J)
    coeffs['W_D3P2_01p'] = _w_coeff(P2, p1, D3, p0, P_J, D_J)
    coeffs['W_D3P2_1p0'] = _w_coeff(P2, p0, D3, p1, P_J, D_J)
    coeffs['W_D3P2_1p1p'] = _w_coeff(P2, p1, D3, p1, P_J, D_J)
    coeffs['W_D3P2_1p2p'] = _w_coeff(P2, p2, D3, p1, P_J, D_J)
    coeffs['W_D3P2_2p1p'] = _w_coeff(P2, p1, D3, p2, P_J, D_J)
    coeffs['W_D3P2_2p2p'] = _w_coeff(P2, p2, D3, p2, P_J, D_J)
    coeffs['W_D3P2_3p2p'] = _w_coeff(P2, p2, D3, p3, P_J, D_J)

    # Initialize Rabi frequencies and detunings to zero
    for trans in ['S1P1', 'S1P2', 'S2P1', 'S2P2', 'D0P1', 'D1P1', 'D1P2', 'D2P1', 'D2P2', 'D3P2']:
        coeffs[f'Omega_{trans}'] = 0.0
        coeffs[f'Delta_{trans}'] = 0.0

    return coeffs


if __name__ == "__main__":
    coeffs = ba137_init()
    print("Ba-137 coefficients initialized successfully.")
    print(f"Number of coefficients: {len(coeffs)}")
    # Print a few sample values
    for key in sorted(coeffs.keys())[:10]:
        print(f"  {key} = {coeffs[key]}")
