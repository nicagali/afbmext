import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from afbm import afbm

sim = afbm(alpha=1, T=10, h=0.01, v=1.0, B_T=1.0, B_R=1.0, KBT=1.0)

r, phi = sim.solve()


fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(r[0], r[1], lw=0.8)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.axis("equal")

fig.savefig('plots/trajectory.png', dpi=200)