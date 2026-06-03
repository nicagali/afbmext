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

def msd(alpha, T, h, v, mu, B_T, B_R, KBT, seed=None):

    if seed is not None:
        np.random.seed(seed)

    sim = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    r, phi = sim.solve()

    dr = r - r[:, [0]]
    dphi = phi - phi[0]

    pos_msd = dr[0]**2 + dr[1]**2
    ang_msd = dphi**2

    return pos_msd, ang_msd

def ensemble_msd(alpha, T, h, v, mu, B_T, B_R, KBT, realizations, n_jobs=-1):

    sim0 = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    n = sim0.n
    t = sim0.t

    seeds = np.random.randint(0, 2**32 - 1, size=realizations)

    results = Parallel(n_jobs=n_jobs)(delayed(msd)(alpha, T, h, v, mu, B_T, B_R, KBT, seed) for seed in tqdm(seeds))

    pos_msd = np.mean([r[0] for r in results], axis=0)
    ang_msd = np.mean([r[1] for r in results], axis=0)

    return t, pos_msd, ang_msd

if __name__ == "__main__":

    t, msd_r, msd_phi = ensemble_msd(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)
    np.savez(f"data/msd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz", t=t, msd_r=msd_r, msd_phi=msd_phi)
