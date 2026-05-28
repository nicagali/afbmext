from fbm import FBM
import numpy as np
from scipy.special import gamma, gammainc
from numba_functions import get_cnj, solver

class afbm:
    def __init__(self, alpha, T, h=0.01, v=1, B_T=1, B_R=1, KBT=1):
        self.alpha = alpha
        self.H = 1 - alpha / 2

        self.T = T
        self.h = h
        self.n = int(T / h)

        self.v = v
        self.KBT = KBT

        self.B_T = B_T
        self.B_R = B_R
        self.A_T = np.sqrt(2 * KBT * B_T / gamma(3 - alpha))
        self.A_R = np.sqrt(2 * KBT * B_R / gamma(3 - alpha))

    def make_B_H(self, method="daviesharte"):
        f = FBM(n=self.n, hurst=self.H, length=self.T, method=method)
        B_H = f.fbm()
        dB_H = np.diff(B_H) * (self.n / self.T)
        return dB_H

    # Prepare the arrays and noise, calculate integration coefficients for the solver
    def prepare(self):
        self.r = np.zeros((2, self.n))
        self.phi = np.zeros(self.n)

        self.coeff_noise = self.h**self.alpha * gamma(2 - self.alpha)

        self.xi_R = self.A_R * self.make_B_H() / self.B_R
        self.xi_T = np.zeros((2, self.n))
        self.xi_T[0] = self.A_T * self.make_B_H() / self.B_T
        self.xi_T[1] = self.A_T * self.make_B_H() / self.B_T

        self.cnj = get_cnj(self.alpha, self.n)

    def solve(self):
        self.prepare()
        solver(self.r,self.phi,self.xi_T,self.xi_R,self.cnj,self.v,self.coeff_noise,self.h,self.n)
        return self.r, self.phi

    # Mean squared analytical
    def get_msd_analytical(self):

        alpha = self.alpha
        v = self.v
        t = np.arange(self.n) * self.h

        DR = self.A_R**2 *gamma(3 - alpha) / (self.B_R**2 * gamma(alpha+1))
        DT = self.A_T**2 *gamma(3 - alpha) / (self.B_T**2 * gamma(alpha+1))

        coeff = ( 2*v**2 / (alpha*((DR)**(2/alpha))))
        gamma1 = gamma(1/alpha)*gammainc(1/alpha, (DR*t**(alpha)))
        gamma2 = gamma(2/alpha)*gammainc(2/alpha, (DR*t**(alpha)))

        self.phi_msd_analytical = DR * t**alpha 
        self.r_msd_analytical = 2*DT * t**alpha + coeff*( gamma1*((DR)**(1/alpha))*t - gamma2  ) 
        