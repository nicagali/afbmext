import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
import numpy as np
plt.style.use("src/plotting_style.mplstyle")

alpha = 0.95

T = 10  
h = 0.001     

v = 200        

a = 1     
B_R = 1
KBT = 1     
B_T = 3 / 4     

mu = a

sim = afbm(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT)

r, phi = sim.solve()
x = r[0]
y = r[1]
t = np.linspace(0, T, len(x))


dt = t[1] - t[0]
phi_mod = np.mod(phi, 2*np.pi)
from scipy.signal import savgol_filter
window = 101   # must be odd
poly = 3
phi_smooth = savgol_filter(phi, window_length=window, polyorder=poly)
omega_smooth = np.gradient(phi_smooth, t)

omega = np.gradient(phi_smooth, dt)
omega_abs = np.abs(np.gradient(phi_smooth, dt))

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(t, phi)
axs[0].plot(t, phi_smooth)
axs[0].set_ylabel(r"$\phi(t)$")

axs[1].plot(t, phi_mod)
axs[1].set_ylabel(r"$\phi(t)$")

axs[2].plot(t, omega)
axs[2].plot(t, omega_abs)
axs[2].set_ylabel(r"$\omega(t)$")

fig.tight_layout()
fig.savefig("plots/single_traj_diagnostics.png", dpi=200)

k = 20
n = len(r[0])
x = np.interp(np.arange(n * k-k), np.arange(n) * k, r[0])
y = np.interp(np.arange(n * k-k), np.arange(n) * k, r[1])
t = np.linspace(0, T, len(x))

import seaborn as sns
cmap = sns.color_palette("Paired", as_cmap=True)
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(x, y, s = 8, c = t, cmap=cmap)
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.axis("equal")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$t/\tau_R$")
fig.savefig(f'plots/traj.png', dpi=200)

# write to xyz for ovito
import plot
plot.npy_to_xyz(filename=f'data/ovito/traj.xyz', r=r, L=a)
