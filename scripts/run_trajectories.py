import numpy as np
from joblib import Parallel, delayed, cpu_count
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

def ensemble_traj(alpha, T, h, v, mu, B_T, B_R, KBT, realizations, sets, start_set, n_jobs=-1):

    sim0 = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    t = sim0.t

    for set_id in tqdm(range(start_set, start_set + sets)):
        
        seeds = np.random.randint(0, 2**32 - 1, size=realizations)

        results = Parallel(n_jobs=n_jobs)(delayed(traj)(alpha, T, h, v, mu, B_T, B_R, KBT, seed) for seed in tqdm(seeds))
            
        r_set = np.array([r[0] for r in results])
        phi_set = np.array([r[1] for r in results])

        np.savez(f"data/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}_set{set_id}.npz", t=t, r_set=r_set, phi_set=phi_set)

    return 

if __name__ == "__main__":

    alpha = float(sys.argv[1])
    T = float(sys.argv[2])
    h = float(sys.argv[3])
    v = float(sys.argv[4])
    mu = float(sys.argv[5])
    realizations = int(sys.argv[6])
    sets = int(sys.argv[7])
    start_set = int(sys.argv[8])
    n_jobs = int(sys.argv[9])
    B_T = 1
    B_R = 1
    KBT = 1

    ensemble_traj(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, sets=sets, start_set=start_set, n_jobs=n_jobs)
