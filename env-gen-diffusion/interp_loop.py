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
from models.viz_utils import save_sampling_video, visualize_elevation_map, visualize_leveled_elevation_map, visualize_elevation_map_pyploy
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
num_samples = 30

elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)
elevation_image = elevation_image[:128, :128]
rows = len(elevation_image)
cols = len(elevation_image[0])

elevation_image = (elevation_image - np.min(elevation_image))
curr_max_x = np.max(elevation_image)
curr_min_x = np.min(elevation_image)
        
elevation = transforms(elevation_image).to(torch.device("cuda"))

masks = []
for i in range(1):
    terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/CA21_1.tif"
    mask_image = np.array(Image.open(terrain_path), dtype=np.float32)[:128, :128]
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

repeated_original_data_sample = elevation.unsqueeze(dim=0).repeat(num_samples, 1, 1, 1)
# we add noises to the denoised sample

for w in np.arange(0.1, 1.0, 0.1):
    # 999 because 1000 cannot work
    ts = [125, 250, 375, 500, 625, 750, 875, 999]
    for i in range(len(ts)):
        edited_x = editor(repeated_original_data_sample.clone(), ts[i], keep_intermediate=False, mask=masks[0].clone(), lamb=w)
        vis_samples = []
        for x in edited_x:
            vis_samples.append(x.cpu().detach().numpy().reshape(rows, cols))

        save_sampling_video(np.array(vis_samples), \
        "test_imgs/step" + str(ts[i]) + "_w_" + str(int(w*10))+".mp4", \
        len(vis_samples)
        )