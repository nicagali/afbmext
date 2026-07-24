import numpy as np

alpha = 0.7
T = 100.
h = 0.05
v = 10.0
mu = 1.
B_T = 1
B_R = 1
KBT = 1
realizations = 600

sets = ["01", "02", "03"]

DATA_PATH = f"data/traj_a{alpha}_T{T}_h{h}_v{v}_mu{mu}_r{realizations}_01.npz"
data = np.load(DATA_PATH)
