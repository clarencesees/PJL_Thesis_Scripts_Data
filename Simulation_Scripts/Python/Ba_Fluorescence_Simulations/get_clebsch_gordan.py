import numpy as np


def get_clebsch_gordan(J1, J2):
    """
    Input definition:
    J1: Angular momentum quantum number of first component
    J2: Angular momentum quantum number of second component
    Output definition:
    A: 2-dimensional matrix of size (2*J1+1)*(2*J2+1)x(2*J1+1)*(2*J2+1). The
    matrix containing all the Clebsch-Gordan coefficients. Rows correspond to
    the state |m1>|m2> while columns correspond to the state <j|<m|. v=A*u
    transforms the u vector in the |jm> basis to the |m1,m2> basis.
    J_index: 1-dimensional array of indices of |j> state
    Jm_index: 1-dimensional array of indices of |m> state
    M1_index: 1-dimensional array of z projection of angular momentum of first component
    M2_index: 1-dimensional array of z projection of angular momentum of second component
    """
    J1 = float(J1)
    J2 = float(J2)
    J = J1 + J2
    J_array = np.arange(J, abs(J1 - J2) - 0.5, -1)
    J_array = J_array[J_array >= abs(J1 - J2)]
    J1_m = np.arange(J1, -J1 - 0.5, -1)
    J2_m = np.arange(J2, -J2 - 0.5, -1)

    dim = int((2 * J1 + 1) * (2 * J2 + 1))
    A = np.full((dim, dim), np.nan)

    # Lowering operators
    n1 = len(J1_m)
    J1_Lower = np.zeros((n1, n1))
    for i in range(n1 - 1):
        J1_Lower[i + 1, i] = np.sqrt(J1 * (J1 + 1) - J1_m[i] * (J1_m[i] - 1))

    n2 = len(J2_m)
    J2_Lower = np.zeros((n2, n2))
    for i in range(n2 - 1):
        J2_Lower[i + 1, i] = np.sqrt(J2 * (J2 + 1) - J2_m[i] * (J2_m[i] - 1))

    J1_I = np.eye(n1)
    J2_I = np.eye(n2)

    Jm_index = np.full(dim, np.nan)
    J_index = np.full(dim, np.nan)

    Jm_length = 0  # 0-indexed
    for h in range(len(J_array)):
        jval = J_array[h]
        n_states = int(2 * jval + 1)
        jm_vals = np.arange(jval, -jval - 0.5, -1)
        jm_vals = jm_vals[:n_states]
        Jm_index[Jm_length:Jm_length + n_states] = jm_vals
        J_index[Jm_length:Jm_length + n_states] = jval
        Jm_length += n_states

    M1_index = np.kron(J1_m, np.ones(n2))
    M2_index = np.kron(np.ones(n1), J2_m)

    Jm_length = 0
    for hh_idx in range(len(J_array)):
        hh = J_array[hh_idx]
        if hh == J:  # |m1=m1_max, m2=m2_max> = |j=j_max, m=m_max>
            v = np.zeros(dim)
            v[0] = 1.0
            A[:, 0] = v
        else:
            # Compute transformation for |j!=j_max, m=j> using simultaneous equation solver
            v = np.zeros(dim)
            col_mask = (J_index > hh) & (np.abs(Jm_index - hh) < 1e-10)
            V = A[:, col_mask].T.copy()
            row_mask = np.abs(M1_index + M2_index - hh) < 1e-10
            V = V[:, row_mask]
            B = -V[:, 0].copy()
            V_reduced = V[:, 1:]
            if V_reduced.size > 0:
                S = np.linalg.solve(V_reduced, B)
                S = np.concatenate(([1.0], S))
            else:
                S = np.array([1.0])
            S = S / np.linalg.norm(S)
            v[row_mask] = S
            A[:, Jm_length] = v

        n_states = int(2 * hh + 1)
        for h in range(Jm_length + 1, Jm_length + n_states):
            # |j, m-1> = N*(J1_- + J2_-)|j, m>, N is a normalizing factor
            lowering_op = np.kron(J1_Lower, J2_I) + np.kron(J1_I, J2_Lower)
            result = lowering_op @ A[:, h - 1]
            norm_val = np.linalg.norm(result)
            if norm_val > 0:
                A[:, h] = result / norm_val
            else:
                A[:, h] = result

        Jm_length += n_states

    return A, J_index, Jm_index, M1_index, M2_index
