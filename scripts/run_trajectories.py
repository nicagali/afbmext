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

def ensemble_traj(alpha, T, h, v, mu, B_T, B_R, KBT, realizations, n_jobs=-1):

    sim0 = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    t = sim0.t

    seeds = np.random.randint(0, 2**32 - 1, size=realizations)

    batch_size = cpu_count() if n_jobs==-1 else n_jobs

    for batch_id, start in enumerate(range(0, realizations, batch_size)):

        end = min(start + batch_size, realizations) #if realizations not multiple of batch size, the last batch will have a different end
        results = Parallel(n_jobs=n_jobs)(delayed(traj)(alpha, T, h, v, mu, B_T, B_R, KBT, seed) for seed in tqdm(seeds[start:end], desc=f"Batch {batch_id + 1}"))

        r_batch = np.array([r[0] for r in results])
        phi_batch = np.array([r[1] for r in results])

        np.savez(f"data/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}_batch{batch_id:02d}.npz", t=t, r_batch=r_batch, phi_batch=phi_batch)

    return 

if __name__ == "__main__":

    alpha = float(sys.argv[1])
    T = float(sys.argv[2])
    h = float(sys.argv[3])
    v = float(sys.argv[4])
    mu = float(sys.argv[5])
    realizations = int(sys.argv[6])
    n_jobs = int(sys.argv[7])
    B_T = 1
    B_R = 1
    KBT = 1

    ensemble_traj(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=n_jobs)
