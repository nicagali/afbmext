from fbm import FBM
import numpy as np
from scipy.special import gamma
from numba import njit


@njit
def get_cnj(alpha, n):
    cnj = np.zeros(n)
    cnj[0] = 1 # We're using the rule 0**x = 0, for any x including 0
    for k in range(1, n):
        cnj[k] = (k + 1)**(1 - alpha) - k**(1 - alpha)
    return cnj

@njit
def solve_phi_no_torque(k, phi, xi_R, cnj, coeff_noise):
    sum_phi = 0.0

    for i in range(1, k):
        sum_phi += cnj[k - i] * (phi[i] - phi[i - 1])

    return phi[k - 1] + coeff_noise * xi_R[k] - sum_phi


@njit
def solve_phi_torque(k, phi, v, xi_R, cnj, h, coeff_noise, torque_strength, max_iter=100, tol=1e-10 ):

    phi_prev = phi[k - 1]
    phi_trial = phi_prev

    # rotational memory term
    sum_phi = 0.0
    for i in range(1, k):
        sum_phi += cnj[k - i] * (phi[i] - phi[i - 1])

    noise = coeff_noise * xi_R[k]

    for _ in range(max_iter):

        torque = 0.0
        dtorque = 0.0

        for i in range(1, k):
            phi_i = phi[i]

            # sin(phi_i - phi_trial)
            torque += cnj[k - i] * ( np.cos(phi_trial) * np.sin(phi_i) - np.sin(phi_trial) * np.cos(phi_i) )

            # derivative of sin(phi_i - phi_trial)
            dtorque += cnj[k - i] * ( -np.sin(phi_trial) * np.sin(phi_i) -np.cos(phi_trial) * np.cos(phi_i) )

        F = ( phi_trial - phi_prev + sum_phi + torque_strength * v * torque * h - noise )

        dF = 1.0 + torque_strength * v * dtorque * h

        step = F / dF
        phi_trial -= step

        if np.abs(step) < tol:
            break

    return phi_trial

@njit
def solver(r, phi, xi_T, xi_R, cnj, v, mu, coeff_noise, h, n):

    # Initialize mx and my for the active term
    mx = np.zeros(n)
    my = np.zeros(n)
    mx[0] = v[0] * np.cos(phi[0]) * h
    my[0] = v[0] * np.sin(phi[0]) * h

    # Main loop for the solver
    for k in range(1, n):

        # if mu > 0:
        #     phi[k] = solve_phi_torque(k, phi, v, xi_R, cnj, h, coeff_noise, mu)
        # else:
        phi[k] = solve_phi_no_torque(k, phi, xi_R, cnj, coeff_noise)

        # Active term
        active_x = 0.0
        active_y = 0.0

        for i in range(1, k+1):
            cnj_i = cnj[k - i]
            active_x += cnj_i * v[i] * np.cos(phi[i]) * h
            active_y += cnj_i * v[i] * np.sin(phi[i]) * h


        # Position memory term
        sum_history_x = 0.0
        sum_history_y = 0.0

        for i in range(1, k):
            sum_history_x += cnj[k - i] * (r[0, i] - r[0, i - 1])
            sum_history_y += cnj[k - i] * (r[1, i] - r[1, i - 1])

        r[0, k] = r[0, k - 1] + coeff_noise * xi_T[0, k] - sum_history_x + active_x
        r[1, k] = r[1, k - 1] + coeff_noise * xi_T[1, k] - sum_history_y + active_y