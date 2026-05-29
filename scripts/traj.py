import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
import numpy as np

alpha = 0.8
T = 2
h = 0.001
v = 200
mu = 1

sim = afbm(alpha=alpha, T=T, h=h, v=v, mu=mu, B_T=1.0, B_R=1, KBT=1)

r, phi = sim.solve()
k = 20
n = len(r[0])
x = np.interp(np.arange(n * k-k), np.arange(n) * k, r[0])
y = np.interp(np.arange(n * k-k), np.arange(n) * k, r[1])
t = np.arange(0, len(x))
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y, s = 10, c = t, cmap='Greens')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axis("equal")
# fig.savefig(f'plots/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}.png', dpi=200)
fig.savefig(f'plots/traj.png', dpi=200)