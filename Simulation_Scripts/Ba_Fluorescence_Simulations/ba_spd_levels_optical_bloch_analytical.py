import numpy as np


def ba_spd_levels_optical_bloch_analytical(Delta_S, Delta_D, Omega_S, Omega_D):
    """
    Computes the analytical P state population for the Ba SPD 3-level system.

    Parameters:
        Delta_S: Detuning of S-P transition laser (rad*MHz)
        Delta_D: Detuning of D-P transition laser (rad*MHz)
        Omega_S: Rabi frequency of S-P transition (rad*MHz)
        Omega_D: Rabi frequency of D-P transition (rad*MHz)

    Returns:
        sigma_PP: P state population (complex, take real part)
    """
    gamma_S = 95.3  # Decay rate of P_1/2 state to S_1/2 in MHz
    gamma_D = 31    # Decay rate of P_1/2 state to D_3/2 in MHz

    # Various components to compute the analytical form for sigma_PP
    AA = Omega_S * gamma_D / Omega_D + Omega_D * gamma_S / Omega_S
    BB = Omega_D / (2 * Omega_S) + Omega_S / (2 * Omega_D)

    Bracket1 = ((AA / (2 * (Delta_S - Delta_D))) * (-Omega_D * Delta_S / Omega_S + Omega_S * Delta_D / Omega_D)
                + (BB / (Delta_S - Delta_D)) * (-(Omega_D * Delta_S * gamma_S / Omega_S) + (Omega_S * Delta_D * gamma_D / Omega_D))
                + ((AA * BB) / (4 * (Delta_S - Delta_D)**2)) * (Omega_D**2 + Omega_S**2)
                + (2 * Delta_S**2 * gamma_S) / (Omega_S**2) + (2 * Delta_D**2 * gamma_D) / (Omega_D**2))

    ImgSPPS = 1j * 2 * gamma_S / Omega_S
    ImgDPPD = 1j * 2 * gamma_D / Omega_D

    sigma_PP = 1.0 / (3 + (2 / (gamma_S + gamma_D)) * Bracket1
                       - 1j * ((gamma_S + gamma_D) / (2 * Omega_S)) * ImgSPPS
                       - 1j * ((gamma_S + gamma_D) / (2 * Omega_D)) * ImgDPPD)

    return sigma_PP


if __name__ == "__main__":
    Delta_S = -2 * np.pi * 10
    Delta_D = -2 * np.pi * 40
    Omega_S = 2 * np.pi * 10
    Omega_D = 2 * np.pi * 10
    sigma_PP = ba_spd_levels_optical_bloch_analytical(Delta_S, Delta_D, Omega_S, Omega_D)
    print(f"sigma_PP = {sigma_PP}")
    print(f"Re(sigma_PP) = {sigma_PP.real}")
