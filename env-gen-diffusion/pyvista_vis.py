from pyvista import examples
import numpy as np
import open3d as o3d
import laspy
import glob
import scipy
from PIL import Image
import pyvista as pv
from scipy import interpolate
from cupyx.scipy.interpolate import RegularGridInterpolator
import cupy as cp

terrain_path = "/home/yy/isaacgym/sim-to-real-offroad/assets/tif/CA21_1.tif"

row = 5
col = 5
elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)
elevation_image = elevation_image[:row*128, :col*128]

elevation_image = (elevation_image - np.min(elevation_image)) / 4.

x = cp.arange(0, len(elevation_image[0]), 1) * 0.1
y = cp.arange(0, len(elevation_image), 1) * 0.1
xg, yg = cp.meshgrid(x, y, indexing='ij')

interp = RegularGridInterpolator((x, y), cp.array(elevation_image), bounds_error=False, fill_value=None, method='linear')

xnew = cp.arange(0, len(elevation_image[0]), 0.5) * 0.1
ynew = cp.arange(0, len(elevation_image), 0.5) * 0.1
Xnew, Ynew = cp.meshgrid(xnew, ynew, indexing='ij')
Znew = interp((Xnew, Ynew))
# Create and plot structured grid
grid = pv.StructuredGrid(Xnew.get(), Ynew.get(), Znew.get())
grid['zz'] = (Znew.get() * 100.0).ravel(order='F')
grid.camera_position = 'xy'
grid.plot(scalars='zz', notebook=False, cmap='gist_earth_r', multi_colors=True, eye_dome_lighting=False, hidden_line_removal=True)

# pl = pv.Plotter()
# _ = pl.add_mesh(grid, smooth_shading=True, show_edges=False, roughness=0.0, show_vertices=False, cmap='gist_earth', multi_colors=True, show_scalar_bar=False)
# pl.camera_position = 'xy'
# pl.save_graphic("~/img.svg")