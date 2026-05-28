import numpy as np

def plot_msd(ax, eq, component, data_path):

    # simulation data
    data = np.load(data_path)
    t = data["t"]
    msd_sim = data["msd_r"] if component == "r" else data["msd_phi"]

    # analytical
    eq.get_msd_analytical()
    t_analytical = np.arange(eq.n) * eq.h
    msd_an = eq.r_msd_analytical if component == "r" else eq.phi_msd_analytical

    ax.plot(t, msd_sim, label=fr"$\alpha={eq.alpha}$")
    ax.plot(t_analytical, msd_an, color = 'black', linestyle="--")
    ax.set_xlabel("t")
    ylabel = r'$\langle (\boldsymbol{{r}}(t)-\boldsymbol{{r}}_0)^2 \rangle$' if component == "r" else r'$\langle (\phi(t)-\phi_0)^2 \rangle$'
    ax.set_ylabel(ylabel)
    ax.legend()

