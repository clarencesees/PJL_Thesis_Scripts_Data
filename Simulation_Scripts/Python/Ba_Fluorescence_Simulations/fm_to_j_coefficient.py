import numpy as np
from get_clebsch_gordan import get_clebsch_gordan
from not_wigner_6j import not_wigner_6j


def fm_to_j_coefficient(F1, m1, F2, m2, k, q, J1, J2, I):
    """
    Output definition:
    result: Coefficient that reduces transition matrix element from |F,m_F> to
    |J>. Scalar.

    Input definition:
    F1: F level of state 1. Scalar.
    m1: z-angular momentum of state 1. Scalar.
    F2: F level of state 2. Scalar.
    m2: z-angular momentum of state 2. Scalar.
    k: Tensor rank. 1 for dipole transition. Scalar.
    q: Polarization of perturbation. Scalar.
    J1: J level of state 1. Scalar.
    J2: J level of state 2. Scalar.
    I: Nuclear spin. Scalar.
    """
    # Generate the Clebsch-Gordan matrix
    A, J_index, Jm_index, M1_index, M2_index = get_clebsch_gordan(F2, k)

    # Pick the Clebsch-Gordan coefficient
    row_mask = (np.abs(M1_index - m2) < 1e-10) & (np.abs(M2_index - q) < 1e-10)
    col_mask = (np.abs(J_index - F1) < 1e-10) & (np.abs(Jm_index - m1) < 1e-10)
    C1_vals = A[row_mask][:, col_mask]

    if C1_vals.size == 0:
        C1 = 0.0
    else:
        C1 = C1_vals.item()

    result = C1

    # Reduce the transition matrix element to J form
    result = result * not_wigner_6j(F1, F2, J1, J2, I, k)

    return result
