import numpy as np
from get_clebsch_gordan import get_clebsch_gordan


def not_wigner_6j(J, Jp, J1, J1p, J2, k):
    """
    Output Definition:
    result: The coefficient obtained after multiplying the Wigner-6j term with
    the factor (-1)^(Jp+J1+k+J2)*sqrt((2*Jp+1)(2*J1+1)). Scalar

    Input Definition:
    J: Unreduced angular momentum of state 1. Scalar.
    Jp: Unreduced angular momentum of state 2. Scalar.
    J1: Reduced angular momentum of state 1. Scalar.
    J1p: Reduced angular momentum of state 2. Scalar.
    J2: Angular momentum component that combines with J1 to form J. Scalar
    k: Rank of tensor operator. Scalar
    """
    m = J  # Computed coefficient is independent of z-angular momentum

    # Generate Clebsch-Gordan Matrices
    A1, A1_J_index, A1_Jm_index, A1_M1_index, A1_M2_index = get_clebsch_gordan(J1, J2)
    A2, A2_Jp_index, A2_Jpm_index, A2_M1p_index, A2_M2_index = get_clebsch_gordan(J1p, J2)
    A3, A3_J_index, A3_Jm_index, A3_Mp_index, A3_q_index = get_clebsch_gordan(Jp, k)
    A4, A4_J1_index, A4_J1m_index, A4_M1p_index, A4_q_index = get_clebsch_gordan(J1p, k)

    result = 0.0

    for mp in np.arange(-Jp, Jp + 0.5, 1):
        for q in np.arange(-k, k + 0.5, 1):
            for m1 in np.arange(-J1, J1 + 0.5, 1):
                for m2 in np.arange(-J2, J2 + 0.5, 1):
                    for m1p in np.arange(-J1p, J1p + 0.5, 1):
                        mask1 = (np.abs(A1_M1_index - m1) < 1e-10) & (np.abs(A1_M2_index - m2) < 1e-10)
                        mask1c = (np.abs(A1_J_index - J) < 1e-10) & (np.abs(A1_Jm_index - m) < 1e-10)
                        C1_vals = A1[mask1][:, mask1c]
                        C1 = C1_vals.item() if C1_vals.size > 0 else 0.0

                        mask2 = (np.abs(A2_M1p_index - m1p) < 1e-10) & (np.abs(A2_M2_index - m2) < 1e-10)
                        mask2c = (np.abs(A2_Jp_index - Jp) < 1e-10) & (np.abs(A2_Jpm_index - mp) < 1e-10)
                        C2_vals = A2[mask2][:, mask2c]
                        C2 = C2_vals.item() if C2_vals.size > 0 else 0.0

                        mask3 = (np.abs(A3_Mp_index - mp) < 1e-10) & (np.abs(A3_q_index - q) < 1e-10)
                        mask3c = (np.abs(A3_J_index - J) < 1e-10) & (np.abs(A3_Jm_index - m) < 1e-10)
                        C3_vals = A3[mask3][:, mask3c]
                        C3 = C3_vals.item() if C3_vals.size > 0 else 0.0

                        mask4 = (np.abs(A4_M1p_index - m1p) < 1e-10) & (np.abs(A4_q_index - q) < 1e-10)
                        mask4c = (np.abs(A4_J1_index - J1) < 1e-10) & (np.abs(A4_J1m_index - m1) < 1e-10)
                        C4_vals = A4[mask4][:, mask4c]
                        C4 = C4_vals.item() if C4_vals.size > 0 else 0.0

                        result += C1 * C2 * C3 * C4

    return result
