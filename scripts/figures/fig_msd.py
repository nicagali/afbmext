import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import sys
sys.path.append("src/")
import plot
from afbm import afbm

alpha = 0.9
T = 100
h = 0.001
v = 200
mu = 1
B_T = 1
B_R = 1
KBT = 1
realizations = 100

DATA_PATH = f"data/msd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz"
    
fig, ax = plt.subplots(figsize=(12, 9))
eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
plot.plot_msd(ax=ax,eq=eq, component="r", data_path=DATA_PATH)
ax.set_xscale("log")    
ax.set_yscale("log")
plt.savefig(f"plots/rmsd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.png", dpi=200)

fig, ax = plt.subplots(figsize=(12, 9))
eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
plot.plot_msd(ax=ax,eq=eq, component="phi", data_path=DATA_PATH)
ax.set_xscale("log")    
ax.set_yscale("log")
plt.savefig(f"plots/phimsd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.png", dpi=200)
