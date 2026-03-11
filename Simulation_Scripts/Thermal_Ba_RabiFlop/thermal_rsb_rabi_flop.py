import numpy as np


def thermal_rsb_rabi_flop(nbar, t, Omega, eta):
    """
    Computes the thermal red sideband Rabi flop signal.

    Parameters:
        nbar: Mean phonon number
        t: Time array
        Omega: Rabi frequency
        eta: Lamb-Dicke parameter

    Returns:
        y: Rabi flop signal
    """
    n_array = np.arange(0, 2001)
    log_P_array = n_array * np.log(nbar) - (n_array + 1) * np.log(nbar + 1)
    P_array = np.exp(log_P_array)
    Omega_array = eta * Omega * np.sqrt(n_array)

    t = np.asarray(t).reshape(-1, 1)
    t_array = np.tile(t, (1, len(n_array)))
    P_array = np.tile(P_array, (len(t), 1))
    Omega_array = np.tile(Omega_array, (len(t), 1))

    y_array = P_array * (np.sin(Omega_array * t_array / 2))**2
    y = np.sum(y_array, axis=1)

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nbar = 10
    t = np.linspace(0, 100, 1000)
    Omega = 2 * np.pi * 0.1
    eta = 0.1

    y = thermal_rsb_rabi_flop(nbar, t, Omega, eta)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Excitation probability')
    plt.title('Thermal RSB Rabi Flop')
    plt.show()
