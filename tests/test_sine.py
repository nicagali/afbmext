import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
import numpy as np
plt.style.use("src/plotting_style.mplstyle")

alpha = 0.9

T = 1
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

dphi = (phi[-1] - phi[:-1]) - (1/6)*(phi[-1] - phi[:-1])**3
sin_dphi = np.sin(dphi)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t[1:], dphi, label=r"$\phi(T) - \phi(t) - \frac{1}{6}(\phi(T) - \phi(t))^3$")
ax.plot(t[1:], sin_dphi, ls=':', label=r"$\sin(\phi(T) - \phi(t))$")
ax.set_xlabel(r"$t$")
ax.legend()
plt.savefig("plots/single_traj_dphi.png", dpi=200)