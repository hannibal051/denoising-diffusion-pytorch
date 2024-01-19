from pyvista import examples
import numpy as np
import glob
import scipy
from PIL import Image
import pyvista as pv
import pywt
import matplotlib.pyplot as plt
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


terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/elevation/test_map4.txt"
# terrain_path = "/Users/sp/sp/sim-to-real-offroad/assets/tif/output_hh.tif"

if terrain_path[-3:] == 'tif':
    elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)[:128*4, :128*4]
elif terrain_path[-3:] == 'txt':
    elevation_image = np.loadtxt(terrain_path, dtype=np.float32, delimiter=',')

elevation_image = (elevation_image - np.min(elevation_image))
elevation_image_origin = elevation_image.copy()
# there is a must to normalize image to [-1, 1]
elevation_image = (elevation_image / np.max(elevation_image)) * 2. - 1.

coeffs2 = pywt.dwt2(elevation_image, 'bior6.8') # bior6.8
LL, (LH, HL, HH) = coeffs2
wave = True

if 1:
    if wave:
        # height = LL
        r = len(LL)
        c = len(LL[0])
        height = np.zeros((r*4, c*4))
        height[]
    else:
        height = elevation_image
    rows = len(height)
    cols = len(height[0])
    editor = Editor("/home/fish/terrain/denoising-diffusion-pytorch/results/model-190.pt")
    transforms = transforms.ToTensor()
    elevation = transforms(height).to(torch.device("cuda"))
    ts = torch.tensor([125, 250, 375, 500, 625, 750, 875, 1000], device=torch.device("cuda"))
    repeated_original_data_sample = elevation.unsqueeze(dim=0).repeat(1, 1, 1, 1)
    # we add noises to the denoised sample
    edited_samples = []
    for i in range(ts.shape[0]):
        edited_x = editor(repeated_original_data_sample, ts[i:i+1].item(), keep_intermediate=False)
        edited_samples.append(edited_x[0].cpu().detach().numpy().reshape(rows, cols))
        print("Shape", edited_samples[-1].shape, edited_x.shape)
    vis_samples = edited_samples
    for id in range(len(vis_samples)):
        if wave:
            coeffs2 = [vis_samples[i], (LH, HL, HH)]
            height = pywt.idwt2(coeffs2, 'bior6.8')
            np.savetxt("./test_imgs/test/"+str(id)+".txt", height)
        else:
            height = vis_samples[i]
            np.savetxt("./test_imgs/test/"+str(id)+".txt", height)

x = np.arange(0, len(elevation_image[0]), 1) * 0.1
y = np.arange(0, len(elevation_image), 1) * 0.1
xg, yg = np.meshgrid(x, y, indexing='ij')

# print(elevation_image.shape)
# print(LL.shape, LH.shape, HL.shape, HH.shape)
# print(np.max(elevation_image), np.min(elevation_image))
# print(np.max(LL), np.min(LL))
# print(np.max(LH), np.min(LH))
# print(np.max(HL), np.min(HL))
# print(np.max(HH), np.min(HH), np.var(HH))

# max_HH = np.max(HH)
# min_HH = np.min(HH)
# HH = (HH - min_HH) / (max_HH - min_HH) * 2.0 - 1.0

# max_HL = np.max(HL)
# min_HL = np.min(HL)
# HL = (HL - min_HL) / (max_HL - min_HL) * 2.0 - 1.0

# max_LH = np.max(LH)
# min_LH = np.min(LH)
# LH = (LH - min_LH) / (max_LH - min_LH) * 2.0 - 1.0

beta = 0.0001
delta = (0.02 - beta) / 1000.
steps = 1000
for _ in range(steps):
    # like a compressed image
    LL = np.sqrt(1-beta)*LL + np.random.randn(*LL.shape) * np.sqrt(beta)
    # horizontal features
    # LH = np.sqrt(1-beta)*LH + np.random.randn(*LH.shape) * np.sqrt(beta)
    # vertical features
    # HL = np.sqrt(1-beta)*HL + np.random.randn(*HL.shape) * np.sqrt(beta)
    # diagonal features
    # HH = np.sqrt(1-beta)*HH + np.random.randn(*HH.shape) * np.sqrt(beta)

    beta += delta
    pass

# LH = (LH + 1.0) / 2.0 * (max_LH - min_LH) + min_LH
# HL = (HL + 1.0) / 2.0 * (max_HL - min_HL) + min_HL
# HH = (HH + 1.0) / 2.0 * (max_HH - min_HH) + min_HH
# print(np.max(HH), np.min(HH))

# LL
# LH will add horizontal bumps
# HL will add vertical bumps
# HH will add many bumps uniformly

coeffs2 = [LL, (LH, HL, HH)]
coeffs3 = pywt.idwt2(coeffs2, 'db8')

elevation_image = coeffs3

# beta = 0.001
# for _ in range(steps):
#     elevation_image = np.sqrt(1-beta)*elevation_image + np.random.randn(*elevation_image.shape) * np.sqrt(beta)
#     beta += delta

x = np.arange(0, len(elevation_image[0]), 1) * 0.1
y = np.arange(0, len(elevation_image), 1) * 0.1
xg, yg = np.meshgrid(x, y, indexing='ij')

grid = pv.StructuredGrid(xg, yg, elevation_image * 2)
grid['lidar'] = elevation_image.ravel(order='F')
grid.camera_position = 'xy'
grid.camera_elevation = -45

boring_cmap = plt.cm.get_cmap("viridis")
grid.plot(scalars='lidar', notebook=False, cmap=boring_cmap, multi_colors=True, eye_dome_lighting=False, show_scalar_bar=0)

# pl = pv.Plotter()
# _ = pl.add_mesh(grid, smooth_shading=True, show_edges=False, roughness=0.0, show_vertices=False, cmap='viridis',
# multi_colors=True, show_scalar_bar=False)
# pl.enable_eye_dome_lighting()
# pl.camera_position = 'xy'
# pl.camera.elevation = -60
# pl.save_graphic("./img.svg")