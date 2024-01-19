import laspy
import glob
import scipy
from PIL import Image
import pyvista as pv
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
import cupy as np
import numpy as np
import matplotlib.pyplot as plt

terrain_path = "/home/yy/isaacgym/sim-to-real-offroad/assets/tif/CA21_1.tif"

row = 5
col = 5
elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)
elevation_image = elevation_image[:row*128, :col*128]


elevation_image = (elevation_image - np.min(elevation_image)) / 2.

x = np.arange(0, len(elevation_image[0]), 1) * 0.1
y = np.arange(0, len(elevation_image), 1) * 0.1
xg, yg = np.meshgrid(x, y, indexing='ij')

# sp = np.stack([xg.ravel(), yg.ravel()], -1)
# interp = RBFInterpolator(sp, np.array(elevation_image).ravel(), kernel='thin_plate_spline')

# xnew = np.arange(0, len(elevation_image[0]), 1) * 0.1
# ynew = np.arange(0, len(elevation_image), 1) * 0.1
# Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
# Znew = interp(np.stack([Xnew.ravel(), Ynew.ravel()], -1))

# zz = Znew.get()
# xx = Xnew.get()
# yy = Ynew.get()
# data = []
# for i in range(len(xx)):
#     for j in range(len(xx[0])):
#         data.append([xx[i][j], yy[i][j], zz[i][j]])
# np.savetxt("/home/yy/terrain.txt", data)
    

# Visualize the elevation map
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(xg, yg, elevation_image, cmap="gist_earth_r", linewidth=0, antialiased=False, rcount=1280, ccount=1280, edgecolor='none')
# surface = ax.plot_surface(Xnew, Ynew, Znew.reshape(len(Xnew), len(Xnew[0])), cmap="jet", linewidth=0, antialiased=False, rcount=1280, ccount=1280, edgecolor='none')
ax.view_init(elev=45.0, azim=-90.0)

ax.axis('off')
ax.grid(False)
ax.set_aspect('equal')
# plt.savefig('/home/yy/terrain.svg', transparent=True)
plt.show()