import numpy as np
from scipy.optimize import least_squares


def getfunpara(fun, P0, x, y, P_LB=None, P_UB=None):
    """
    Input definition:
    fun: Function for nonlinear regression. Should have signature fun(x, *params).
    P0: Initial guess of parameters.
    x: x data points.
    y: y data points.
    P_LB: Lower bound of parameters.
    P_UB: Upper bound of parameters.

    Output definition:
    BETA: Converged parameters.
    BETA_Err: Uncertainties of converged parameters.
    Exitflag: Condition that led to termination of regression.
    """
    P0 = np.asarray(P0, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def residuals(params):
        return fun(x, *params) - y

    if P_LB is not None and P_UB is not None:
        bounds = (np.asarray(P_LB, dtype=float), np.asarray(P_UB, dtype=float))
    else:
        bounds = (-np.inf, np.inf)

    result = least_squares(residuals, P0, bounds=bounds)

    BETA = result.x
    Exitflag = result.status

    # Compute parameter uncertainties
    R = result.fun
    J = result.jac
    MSE = np.dot(R, R) / (len(y) - len(P0))
    try:
        COVB = np.linalg.pinv(J.T @ J) * MSE
        BETA_Err = np.sqrt(np.diag(COVB))
    except np.linalg.LinAlgError:
        BETA_Err = np.full(len(P0), np.nan)

    return BETA, BETA_Err, Exitflag
