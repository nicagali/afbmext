import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import sys
sys.path.append("src/")
import plot
from afbm import afbm

alpha=0.5
T=100.
h=0.001
mu=1.
realizations=500
B_T = 3 / 4 
B_R = 1
KBT = 1

fig, ax = plt.subplots(figsize=(9, 7))

add_analytics = True
for v in [0., 10., 50., 100., 200.]:
    DATA_PATH = f"data/msd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz"
    eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    plot.plot_msd(ax=ax,eq=eq, component="r", add_analytics=add_analytics, label=fr"$v\tau_R/a={eq.v}$", data_path=DATA_PATH)
    ax.set_xscale("log")    
    ax.set_yscale("log")
ax.set_xlim(h)
plt.savefig(f"plots/rmsdvarv_a{alpha}_T{T}_h{h}_mu{mu}_r{realizations}.png", dpi=200)

fig, ax = plt.subplots(figsize=(9, 7))
for v in [0., 10., 50., 100., 200.]:

    DATA_PATH = f"data/msd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz"
    eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
    plot.plot_msd(ax=ax,eq=eq, component="phi", add_analytics=add_analytics, label=fr"$v\tau_R/a={eq.v}$", data_path=DATA_PATH)

    ax.set_xscale("log")    
    ax.set_yscale("log")


plot.add_trend(ax, tmin=1, tmax=T, exponent=alpha, yref=1, tref=1, color='blue', linestyle='--')
# plot.add_trend(ax, tmin=0.03, tmax=0.5, exponent=2.9, yref=5e3, tref=1, color='#ff0080', linestyle='--')
# plot.add_trend(ax, tmin=0.01, tmax=0.1, exponent=3.5, yref=5e5, tref=1, color='#bedf1d', linestyle='--')
ax.set_xlim(h)
ax.legend()                      
plt.savefig(f"plots/phimsdvarv_a{alpha}_T{T}_h{h}_mu{mu}_r{realizations}.png", dpi=200)
