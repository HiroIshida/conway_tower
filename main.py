import numpy as np
from build.conway_tower import evolve_conway
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inp = np.zeros((56, 56), dtype=bool)
inp[20:40, 20:40] = np.random.choice([True, False], (20, 20), p=[0.5, 0.5])
t_step = 55
out = evolve_conway(inp, t_step, True)
out = np.moveaxis(out, 0, -1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(out, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time Step')
ax.set_title("Conway's Game of Life - 3D Visualization")
ax.view_init(elev=20, azim=45)
plt.show()
