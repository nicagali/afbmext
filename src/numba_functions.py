from fbm import FBM
import numpy as np
from scipy.special import gamma
from numba import njit


@njit
def get_cnj(alpha, n):
    cnj = np.zeros(n)
    for k in range(1, n):
        cnj[k] = (k + 1)**(1 - alpha) - k**(1 - alpha)
    return cnj


@njit
def solver(r, phi, xi_T, xi_R, cnj, v, coeff_noise, h, n):
    for k in range(1, n):

        # Update angle
        sum_phi = 0.0
        for i in range(1, k):
            sum_phi += cnj[k - i] * (phi[i] - phi[i - 1])

        phi[k] = phi[k - 1] + coeff_noise * xi_R[k] - sum_phi

        # Active term
        sum_nx = 0.0
        sum_ny = 0.0

        for i in range(1, k):
            sum_nx += cnj[k - i] * np.cos(phi[i]) * h
            sum_ny += cnj[k - i] * np.sin(phi[i]) * h

        active_x = v * sum_nx + v * np.cos(phi[k]) * h
        active_y = v * sum_ny + v * np.sin(phi[k]) * h

        # Position memory term
        sum_history_x = 0.0
        sum_history_y = 0.0

        for i in range(1, k):
            sum_history_x += cnj[k - i] * (r[0, i] - r[0, i - 1])
            sum_history_y += cnj[k - i] * (r[1, i] - r[1, i - 1])

        r[0, k] = r[0, k - 1] + coeff_noise * xi_T[0, k] - sum_history_x + active_x
        r[1, k] = r[1, k - 1] + coeff_noise * xi_T[1, k] - sum_history_y + active_y