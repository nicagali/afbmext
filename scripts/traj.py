import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm
import numpy as np

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
k = 20
n = len(r[0])
x = np.interp(np.arange(n * k-k), np.arange(n) * k, r[0])
y = np.interp(np.arange(n * k-k), np.arange(n) * k, r[1])
t = np.linspace(0, T, len(x))
# tstart = 0.1
# tend = 3
# tstart_index = int(T / h * tstart)
# tend_index = int(T / h * tend)
# print(f"tstart_index: {tstart_index}, tend_index: {tend_index}")
# x = x[tstart_index:tend_index]
# y = y[tstart_index:tend_index]
# t = t[tstart_index:tend_index]
# 
import seaborn as sns
cmap = sns.color_palette("Paired", as_cmap=True)
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(x, y, s = 8, c = t, cmap=cmap)
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.axis("equal")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$t/a$")
fig.savefig(f'plots/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}.png', dpi=200)
# fig.savefig(f'plots/traj{v}.png', dpi=200)
# fig.savefig(f'plots/traj.png', dpi=200)