import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import sys
sys.path.append("src/")
import plot
from afbm import afbm

alpha = 0.9
T = 10.
h = 0.01
v = 1.
mu = 1.
B_T = 1
B_R = 1
KBT = 1
realizations = 100

DATA_PATH = f"data/phistats_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz"

data = np.load(DATA_PATH)

t = data["t"]
mean_dphi = data["mean_dphi"]
var_dphi = data["var_dphi"]
mean_omega = data["mean_omega"]


fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axs[0].plot(t, mean_dphi)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_ylabel(r"$\langle\Delta\phi\rangle$")

axs[1].plot(t, var_dphi)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_ylabel(r"Var$[\Delta\phi]$")


axs[2].plot(t, mean_omega)
axs[2].set_xscale("log")
axs[2].set_ylabel(r"$\langle\omega\rangle$")
axs[2].set_xlabel(r"$t$")

plt.tight_layout()

plt.savefig( f"plots/phistats_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.png", dpi=200 )