import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append("src/")
from afbm import afbm
from tqdm import tqdm

def traj(alpha, T, h, v, mu, B_T, B_R, KBT, seed=None):

    if seed is not None:
        np.random.seed(seed)

    sim = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    r, phi = sim.solve()

    return r, phi

def ensemble_traj(alpha, T, h, v, mu, B_T, B_R, KBT, realizations, n_jobs=-1):

    sim0 = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    t = sim0.t

    seeds = np.random.randint(0, 2**32 - 1, size=realizations)

    results = Parallel(n_jobs=n_jobs)(delayed(traj)(alpha, T, h, v, mu, B_T, B_R, KBT, seed) for seed in tqdm(seeds))

    pos_traj = np.array([r[0] for r in results])
    ang_traj = np.array([r[1] for r in results])

    return t, pos_traj, ang_traj

if __name__ == "__main__":

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

    t, traj_r, traj_phi = ensemble_traj(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)
    np.savez(f"data/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz", t=t, traj_r=traj_r, traj_phi=traj_phi)
