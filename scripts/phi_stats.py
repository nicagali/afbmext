import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append("src/")
from afbm import afbm
from tqdm import tqdm

alpha = float(sys.argv[1])
T = float(sys.argv[2])
h = float(sys.argv[3])
v = float(sys.argv[4])
mu = float(sys.argv[5])
realizations = int(sys.argv[6])
B_T = 1
B_R = 1
KBT = 1
n_jobs=-1 #use all cores

def phi_run(alpha, T, h, v, mu, B_T, B_R, KBT, seed=None):

    if seed is not None:
        np.random.seed(seed)

    sim = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    r, phi = sim.solve()

    dphi = phi - phi[0]
    omega = np.gradient(phi, h)

    return dphi, omega

def ensemble_phi(alpha, T, h, v, mu, B_T, B_R, KBT, realizations, n_jobs=-1):

    sim0 = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    t = sim0.t

    seeds = np.random.randint(0, 2**32 - 1, size=realizations)

    results = Parallel(n_jobs=n_jobs)(
        delayed(phi_run)(alpha, T, h, v, mu, B_T, B_R, KBT, seed)
        for seed in tqdm(seeds)
    )

    dphis = np.array([res[0] for res in results])
    omegas = np.array([res[1] for res in results])

    # Mean angular displacement
    mean_dphi = np.mean(dphis, axis=0)

    # Angular variance
    var_dphi = np.mean(dphis**2, axis=0) - mean_dphi**2

    # Mean angular velocity
    mean_omega = np.mean(omegas, axis=0)

    return t, mean_dphi, var_dphi, mean_omega
if __name__ == "__main__":

    t, mean_dphi, var_dphi, mean_omega = ensemble_phi(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)
    np.savez(f"data/phistats_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz", t=t, mean_dphi=mean_dphi, var_dphi=var_dphi, mean_omega=mean_omega)
