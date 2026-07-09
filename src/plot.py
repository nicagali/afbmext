import numpy as np

def plot_msd(ax, eq, component, add_analytics, label, data_path, ylabel = True, xlabel = True):

    # simulation data
    data = np.load(data_path)
    t = data["t"]
    msd_sim = data["msd_r"] if component == "r" else data["msd_phi"]

    # analytical
    eq.get_msd_analytical()
    t_analytical = np.arange(eq.n) * eq.h
    msd_an = eq.r_msd_analytical if component == "r" else eq.phi_msd_analytical

    ax.plot(t, msd_sim, label=label)
    if add_analytics:
        ax.plot(t_analytical, msd_an, color = 'black', linestyle="--")
    if xlabel:
        ax.set_xlabel(r"$t/\tau_R$")
    if ylabel:
        ylabel = r'$\langle (\boldsymbol{{r}}(t)-\boldsymbol{{r}}_0)^2 \rangle/a^2$' if component == "r" else r'$\langle (\phi(t)-\phi_0)^2 \rangle$'
        ax.set_ylabel(ylabel)
    ax.legend()

def add_trend(ax, tmin, tmax, exponent, yref=1, tref=1, **kwargs):

    t = np.arange(tmin,tmax,0.0001)

    y = yref * (t / tref)**exponent

    ax.plot(t, y, label=rf'$~t^{{{exponent}}}$', **kwargs)


def npy_to_xyz(filename, r, L):
    n = r.shape[1]

    with open(filename, "w") as f:
        for t in range(n):
            # number of atoms
            f.write("1\n")

            # comment / metadata line
            f.write(
                f'Lattice="{L} 0 0 0 {L} 0 0 0 1.0" '
                f'Properties=species:S:1:pos:R:3 '
                f'Frame={t}\n'
            )

            x = r[0, t]
            y = r[1, t]
            f.write(f"A {x} {y} 0.0\n")