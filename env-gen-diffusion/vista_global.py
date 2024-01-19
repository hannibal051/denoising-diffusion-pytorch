from pyvista import examples
import numpy as np
import laspy
# import glob
import scipy
from PIL import Image
import pyvista as pv
from scipy import interpolate
# from cupyx.scipy.interpolate import RegularGridInterpolator
# import cupy as cp
import matplotlib.pyplot as plt
# from models.viz_utils import save_sampling_video, visualize_elevation_map, visualize_leveled_elevation_map, visualize_elevation_map_pyploy
# from data_script.elevation_dataset import ElevationDataset
from models.ddpm_edit import Editor
import torch
# from models.util import ScaleToRange, scale_to_range
import numpy as np
import torchvision.transforms as transforms

editor = Editor("/home/fish/terrain/denoising-diffusion-pytorch/results/model-190.pt")

def scale_to_range(x, curr_min_x, curr_max_x, min_x_new, max_x_new):
    """
    This function is useful for scaling the input of the DDPM to the range of [-1, 1],
    and the output of the DDPM to the range of [0, 1] 
    """
    return (x - curr_min_x) * (max_x_new - min_x_new) / (curr_max_x - curr_min_x) + min_x_new

transforms = transforms.ToTensor()
# terrain_path = "/home/fish/test/output_hh_0.tif"
terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/output_hh.tif"
num_samples = 1

elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)
elevation_image = elevation_image[:128*4, :128*4]
rows = len(elevation_image)
cols = len(elevation_image[0])

elevation_image = (elevation_image - np.min(elevation_image))
curr_max_x = np.max(elevation_image)
curr_min_x = np.min(elevation_image)
        
elevation = transforms(elevation_image).to(torch.device("cuda"))

use_mask = True
if use_mask:
    masks = []
    for i in range(1, 8):
        terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/output_hh.tif"
        mask_image = np.array(Image.open(terrain_path), dtype=np.float32)

        x_min = min((i // rows) * rows, mask_image.shape[0] - rows)
        y_min = min((i % rows) * cols, mask_image.shape[1] - cols) 
        x_max = x_min + rows
        y_max = y_min + cols
        # mask_image = mask_image[x_min:x_max, y_min:y_max]
        mask_image = mask_image[128*4:128*4+128*4, :128*4]
        curr_max_x = np.max(mask_image)
        curr_min_x = np.min(mask_image)
        mask = transforms(mask_image).to(torch.device("cuda"))
        max_x_new = 1.0
        min_x_new = -1.0
        mask = scale_to_range(mask, curr_min_x, curr_max_x, min_x_new, max_x_new)
        masks.append(mask.view(1, 1, rows, cols).repeat(num_samples, 1, 1, 1))

# after the totensor transform, the image range is within [0, 1]
# we now need to scale it to range [-1, 1] for DDPM 
max_x_new = 1.0
min_x_new = -1.0
elevation = scale_to_range(elevation, curr_min_x, curr_max_x, min_x_new, max_x_new)
# choose 6 different forward steps
ts = torch.tensor([125, 250, 375, 500, 625, 750, 875], device=torch.device("cuda"))
if use_mask:
    ts = torch.tensor([875, 875, 875, 875, 875, 875, 875], device=torch.device("cuda"))
# every diffusion step is used to generate 5 samples to test the diversity 

repeated_original_data_sample = elevation.unsqueeze(dim=0).repeat(num_samples, 1, 1, 1)
# we add noises to the denoised sample
edited_samples = []
for i in range(ts.shape[0]):
    if use_mask:
        edited_x = editor(repeated_original_data_sample, ts[i:i+1].item(), keep_intermediate=False, mask=masks[i], lamb=0.9 - 0.1*i)
    else:
        edited_x = editor(repeated_original_data_sample, ts[i:i+1].item(), keep_intermediate=False)
    edited_samples.append(edited_x[0].cpu().detach().numpy().reshape(rows, cols))
    print("Shape", edited_samples[-1].shape, edited_x.shape)

vis_samples = edited_samples



for id in range(len(vis_samples)):
    elevation_image = vis_samples[i]

    np.savetxt("./test_imgs/test/"+str(id)+".txt", elevation_image)
    continue
    x = np.arange(0, len(elevation_image[0]), 1) * 0.1
    y = np.arange(0, len(elevation_image), 1) * 0.1
    xg, yg = np.meshgrid(x, y, indexing='ij')

    # interp = RegularGridInterpolator((x, y), cp.array(vis_samples), bounds_error=False, fill_value=None, method='linear')

    # xnew = cp.arange(0, len(elevation_image[0]), 1) * 0.1
    # ynew = cp.arange(0, len(elevation_image), 1) * 0.1
    # Xnew, Ynew = cp.meshgrid(xnew, ynew, indexing='ij')
    # Znew = interp((Xnew, Ynew))
    # Create and plot structured grid
    pv.start_xvfb()

    grid = pv.StructuredGrid(xg, yg, elevation_image)
    grid['lidar'] = elevation_image.ravel(order='F')
    grid.camera_position = 'xy'
    # grid.plot(scalars='lidar', notebook=False, cmap='gist_earth', multi_colors=True, eye_dome_lighting=True, hidden_line_removal=True)

    pl = pv.Plotter()
    _ = pl.add_mesh(grid, smooth_shading=True, show_edges=False, show_vertices=False, cmap='gist_earth', multi_colors=True)
    # pl.camera_position = 'xy'
    pl.save_graphic("./test_imgs/"+str(id)+".svg")