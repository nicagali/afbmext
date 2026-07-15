import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import sys
sys.path.append("scripts/")
from msd import ensemble_msd
sys.path.append("src/")
from afbm import afbm
import numpy as np

alpha = 0.9

T = 2

h = 0.005   
n = int(T / h)

a = 1     
B_R = 1
KBT = 1     
B_T = 3 / 4     

mu = a

realizations = 500

seed = 12345
np.random.seed(seed)
fig, ax = plt.subplots(figsize=(7, 7))
figmsd, axmsd = plt.subplots(figsize=(7, 7))
figphimsd, axphimsd = plt.subplots(figsize=(7, 7))

#---------------- constant velocity 
v = 200.0
sim = afbm(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT)
r, _ = sim.solve()
t, msd_r, msd_phi = ensemble_msd(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)

axmsd.plot(t, msd_r, c="#FE6C23", label=fr"$v\tau_R/a={v}$")
axphimsd.plot(t, msd_phi, c="#FE6C23", label=fr"$v\tau_R/a={v}$")

k = 20
x = np.interp(np.arange(n * k-k), np.arange(n) * k, r[0])
y = np.interp(np.arange(n * k-k), np.arange(n) * k, r[1])
t = np.linspace(0, T, len(x))

ax.scatter(x, y, s=8, label=fr"$v\tau_R/a={v}$")

#---------------- intermittent velocity
np.random.seed(seed)
v1= 200.0
v2= 0.0
def v_periodic_bloks(n, peak, v1, v2):

    block_size_v1 = int(0.5*peak)
    block_size_v2 = int(0.2*peak)

    cycle_size = block_size_v1 + block_size_v2
    position_in_cycle = np.arange(n) % cycle_size

    v = np.where(position_in_cycle < block_size_v1, v1, v2)

    return v

def v_three_blocks(n, peak, v1, v2):
    block_size_v1 = int(0.5 * peak)
    block_size_v2 = int(0.5 * peak)

    v = np.full(n, v1, dtype=np.float64)

    start_v2 = block_size_v1
    end_v2 = min(start_v2 + block_size_v2, n)

    # v[start_v2:end_v2] = v2
    v[:end_v2] = v2

    return v

v = v_periodic_bloks(n, peak=50, v1=v1, v2=v2)
# v = v_three_blocks(n, peak=500, v1=v1, v2=v2)

sim = afbm(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT)
r, _ = sim.solve()
t, msd_r, msd_phi = ensemble_msd(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)

axmsd.plot(t, msd_r, label=fr"$v\tau_R/a={v1}$ and $v\tau_R/a={v2}$")
axphimsd.plot(t, msd_phi, label=fr"$v\tau_R/a={v1}$ and $v\tau_R/a={v2}$")

change_idx = np.where(v[1:] != v[:-1])[0] + 1
change_idx = change_idx[change_idx < len(t)]
axmsd.scatter( t[change_idx], msd_r[change_idx], s=45, color="black", marker="o", label="velocity change", zorder=5)

# tajectory
k = 20
x = np.interp(np.arange(n * k-k), np.arange(n) * k, r[0])
y = np.interp(np.arange(n * k-k), np.arange(n) * k, r[1])
t = np.linspace(0, T, len(x))

# Interpolate velocity as well
v_plot = np.repeat(v[:-1], k)
# find the indices where v_plot is equal to v1 and v2
mask_v1 = np.isclose(v_plot, v1)
mask_v2 = np.isclose(v_plot, v2)

scatter = ax.scatter(x[mask_v1], y[mask_v1], s=8, color='red', label=fr"$v\tau_R/a={v1}$")
scatter = ax.scatter(x[mask_v2], y[mask_v2], s=8, color='blue', label=fr"$v\tau_R/a={v2}$")

# ----------------
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.axis("equal")
fig.savefig(f'plots/traj.png', dpi=200)

axmsd.set_xlabel(r"$t/\tau_R$")
axmsd.set_ylabel(r"$\langle |\boldsymbol{r}(t)-\boldsymbol{r}(0)|^2\rangle/a^2$")
axmsd.legend()
figmsd.savefig(f'plots/msd.png', dpi=200)

axphimsd.set_xlabel(r"$t/\tau_R$")
axphimsd.set_ylabel(r"$\langle |\phi(t)-\phi(0)|^2\rangle$")
axphimsd.legend()
figphimsd.savefig(f'plots/msd_phi.png', dpi=200)