import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append("src/")
from afbm import afbm
from tqdm import tqdm


def ensemble_msd(alpha, T, h, v, mu, sets, start_set, realizations):

    n = int(T/h)
    sum_msd_r = np.zeros(n)
    sum_msd_phi = np.zeros(n)
    total_realizations = sets*realizations

    for set_id in range(start_set, start_set + sets):

        data = np.load(f"data/traj_a{alpha}_T{T}_h{h}_v{v}_" f"mu{mu}_r{realizations}_set{set_id}.npz")

        t = data["t"]
        r = data["r_set"]          # shape: (realizations, 2, n)
        phi = data["phi_set"]      # shape: (realizations, n)

        dr = r - r[:, :, [0]]
        dphi = phi - phi[:, [0]]

        msd_r_set = np.sum(dr**2, axis=1)  # shape: (realizations, n)
        msd_phi_set = dphi**2                # shape: (realizations, n)

        sum_msd_r += np.sum(msd_r_set, axis=0)
        sum_msd_phi += np.sum(msd_phi_set, axis=0)

    msd_r = sum_msd_r / total_realizations
    msd_phi = sum_msd_phi / total_realizations

    return t, msd_r, msd_phi

if __name__ == "__main__":

    alpha = 0.9
    T = 100.
    h = 0.01
    v = 200.
    mu = 1.
    realizations = 100
    sets = 2
    start_set = 0
    B_T = 1
    B_R = 1
    KBT = 1

    t, msd_r, msd_phi = ensemble_msd( alpha=alpha, T=T, h=h, v=v, mu=mu, sets=sets, start_set=start_set, realizations=realizations )

    np.savez(f"data/msd_a{alpha}_T{T}_h{h}_v{v}_" f"mu{mu}_r{realizations}_sets{sets}.npz", t=t, msd_r=msd_r, msd_phi=msd_phi)