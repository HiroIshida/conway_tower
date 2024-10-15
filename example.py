import numpy as np
from conway_tower import evolve_conway
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

init_state = np.zeros((60, 60), dtype=bool)
init_state[20:40, 20:40] = np.random.choice([True, False], (20, 20), p=[0.5, 0.5])
t_step = 60

ts = time.time()
out = evolve_conway(init_state, t_step, z_as_time=True)
print(f"time to build: {(time.time() - ts)* 1000} ms")

print("visualizing voxels...")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(out, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time Step')
ax.set_title("Conway's Game of Life - 3D Visualization")
ax.view_init(elev=20, azim=45)
plt.show()
