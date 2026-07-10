import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
import numpy as np
plt.style.use("src/plotting_style.mplstyle")

alpha = 0.7

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

dphi = np.diff(phi)
sin_dphi = np.sin(dphi)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t[1:], dphi, label=r"$\phi(t) - \phi(t-\Delta t)$")
ax.plot(t[1:], sin_dphi, label=r"$\sin(\phi(t) - \phi(t-\Delta t))$")
ax.set_xlabel(r"$t$")
ax.legend()
plt.savefig("plots/single_traj_dphi.png", dpi=200)