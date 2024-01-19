import numpy as np
import open3d as o3d
import laspy
import glob
import scipy
from PIL import Image


terrain_path = "/home/yy/isaacgym/sim-to-real-offroad/assets/tif/CA21_1.tif"

row = 10
col = 10
elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)
elevation_image = elevation_image[:row*128, :col*128]


elevation_image = (elevation_image - np.min(elevation_image)) / 3.

x = np.arange(0, len(elevation_image[0]), 1)
y = np.arange(0, len(elevation_image), 1)
X, Y = np.meshgrid(x, y)

# interp = scipy.interpolate.RegularGridInterpolator((x, y), elevation_image)
# xnew = np.arange(0, len(elevation_image[0])-1, 0.5)
# ynew = np.arange(0, len(elevation_image)-1, 0.5)
# xnew, ynew = np.meshgrid(xnew, ynew)
# test_points = np.array([xnew.ravel(), ynew.ravel()]).T
# znew = interp(test_points, method='cubic')
# xnew, ynew = np.meshgrid(xnew, ynew)

points = np.hstack((X.reshape(-1,1) * 0.1, Y.reshape(-1,1) * 0.1, elevation_image.reshape(-1,1)))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

colors = np.zeros_like(points)
colors[:, 0] = np.ones_like(colors[:, 0]) * 108. / 255.
colors[:, 1] = np.ones_like(colors[:, 0]) * 116. / 255.
colors[:, 2] = np.ones_like(colors[:, 0]) * 118. / 255.
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])