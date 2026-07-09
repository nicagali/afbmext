import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import sys
sys.path.append("src/")
import plot
from afbm import afbm

alphas = [0.9, 0.7, 0.5]
T=100.
h=0.001
mu=1.
realizations=500
B_T = 3 / 4 
B_R = 1
KBT = 1

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

for a, alpha in enumerate(alphas):
    for j, component in enumerate(['phi', 'r']):  # replace with your existing second loop / condition

        ax = axes[j, a]

        ylabel = True if a == 0 else False
        xlabel = True if j == 1 else False

        add_analytics = False
        for v in [0., 10., 50., 100., 200.]:
            DATA_PATH = f"data/msd_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}.npz"
            eq = afbm(alpha, T, h, v, mu, B_T, B_R, KBT)
            plot.plot_msd(ax=ax,eq=eq, component=component, add_analytics=add_analytics, label=fr"$v\tau_R/a={eq.v}$", data_path=DATA_PATH, ylabel = ylabel, xlabel = xlabel)
            ax.set_xscale("log")    
            ax.set_yscale("log")

        ax.get_legend().remove()
        ax.set_xlim(h)
    axes[0, a].set_title(f"alpha = {alpha}")
    handles, legend = ax.get_legend_handles_labels()

fig.legend(handles, legend, bbox_to_anchor = (1,1.1), ncol = 5)

plt.savefig(f"plots/rmsdvarv_a{alphas}_T{T}_h{h}_mu{mu}_r{realizations}.png", dpi=200)
