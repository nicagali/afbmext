import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import sys
sys.path.append("src/")
import plot
from afbm import afbm

alpha = 0.7
T = 20
h = 0.001
v = 0
B_T = 1
B_R = 1
KBT = 1
realizations = 100

DATA_PATH = f"data/msd_a{alpha}_T{T}_h{h}_v{v}_r{realizations}.npz"
    
fig, ax = plt.subplots(figsize=(12, 9))
eq = afbm(alpha, T, h, v, B_T, B_R, KBT)
plot.plot_msd(ax=ax,eq=eq, component="r", data_path=DATA_PATH)
plt.savefig(f"plots/rmsd_a{alpha}_T{T}_h{h}_v{v}_r{realizations}.png", dpi=200)

fig, ax = plt.subplots(figsize=(12, 9))
eq = afbm(alpha, T, h, v, B_T, B_R, KBT)
plot.plot_msd(ax=ax,eq=eq, component="phi", data_path=DATA_PATH)
plt.savefig(f"plots/phimsd_a{alpha}_T{T}_h{h}_v{v}_r{realizations}.png", dpi=200)
