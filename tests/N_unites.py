import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
sys.path.append("scripts/")
from msd import ensemble_msd
import numpy as np
plt.style.use("src/plotting_style.mplstyle")
# This tests gives dimensions of Narinder's paper to the algorithm

alpha = 1

T = 10        # [s]
h = 0.1      # [s]

v = 0          # [um/s]

a = 3.875      # [um]
eta = 0.004    # [pN s / um^2]

B_T = 6*np.pi*a*eta          # [pN s / um]
B_R = 8*np.pi*(a**3)*eta     # [pN um s]

KBT = (1.38e-23)*(298)*1e18  # [pN um]

mu = a

realizations = 100
t, msd_r, msd_phi = ensemble_msd(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=B_T, B_R=B_R, KBT=KBT, realizations=realizations, n_jobs=-1)

D = KBT/(B_R)
print(D)
msd_an = 2*D*t

eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
eq.get_msd_analytical()
t_analytical = np.arange(eq.n) * eq.h
rmsd_an = eq.r_msd_analytical 
phimsd_an =eq.phi_msd_analytical

print(msd_phi[0], msd_phi[1])
print(msd_an[0], msd_an[1])
fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(t, msd_phi)
ax.plot(t, msd_an)
ax.set_xlim(h)
# ax.plot(t, phimsd_an)
ax.set_xscale("log")    
ax.set_yscale("log")
plt.savefig(f"plots/phimsd_Nunits.png", dpi=200)


# fig, ax = plt.subplots(figsize=(12, 9))
# ax.plot(t, msd_r)
# ax.plot(t, rmsd_an)
# ax.set_xscale("log")    
# ax.set_yscale("log")
# plt.savefig(f"plots/rmsd_Nunits.png", dpi=200)
