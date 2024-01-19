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

import random
from PIL import Image, ImageDraw


def check_overlap(mask, shape_bbox):
    x1, y1, x2, y2 = shape_bbox
    subarea = mask.crop((x1, y1, x2, y2))
    return subarea.getbbox() is not None


def generate_mask(H, W, mask_area_ratio, elevation):
    mask = Image.new('1', (W, H), 0)
    # mask = Image.fromarray(elevation.cpu().numpy()[0])
    draw = ImageDraw.Draw(mask)

    target_area = H * W * mask_area_ratio
    current_area = 0

    while True:
        shape = random.choice(['circle', 'rectangle', 'polygon'])
        new_area = 0
        bbox = ()

        if shape == 'circle':
            x1, y1 = random.randint(0, W), random.randint(0, H)
            r = random.randint(1, min(W, H) // 6)
            bbox = (x1 - r, y1 - r, x1 + r, y1 + r)
            new_area = round(3.14159 * r * r)

        elif shape == 'rectangle':
            x1, y1 = random.randint(0, W//2), random.randint(0, H//2)
            x2, y2 = random.randint(x1, W//2), random.randint(y1, H//2)
            bbox = (x1, y1, x2, y2)
            new_area = abs((x2 - x1) * (y2 - y1))

        elif shape == 'polygon':
            points_num = random.randint(3, 6)
            points = [(random.randint(0, W//2), random.randint(0, H//2)) for _ in range(points_num)]
            bbox = tuple(map(min, zip(*points))) + tuple(map(max, zip(*points)))

            # Calculate the area of the polygon using Shoelace formula
            x, y = zip(*points)
            new_area = int(0.5 * abs(sum(x[i - 1] * y[i] - x[i] * y[i - 1] for i in range(points_num))))
        if current_area + new_area <= target_area:
            if shape == 'circle':
                draw.ellipse(bbox, fill=1)
            elif shape == 'rectangle':
                draw.rectangle(bbox, fill=1)
            elif shape == 'polygon':
                draw.polygon(points, fill=1)
            current_area += new_area
        else:
            break

        if current_area >= target_area:
            break
    return mask

editor = Editor("/home/fish/terrain/denoising-diffusion-pytorch/results/model-190.pt")
num_samples = 1

def scale_to_range(x, curr_min_x, curr_max_x, min_x_new, max_x_new):
    """
    This function is useful for scaling the input of the DDPM to the range of [-1, 1],
    and the output of the DDPM to the range of [0, 1] 
    """
    return (x - curr_min_x) * (max_x_new - min_x_new) / (curr_max_x - curr_min_x) + min_x_new

transforms = transforms.ToTensor()
# terrain_path = "/home/fish/test/output_hh_0.tif"
terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/output_hh.tif"

elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)[:128*4, :128*4]
elevation_image = elevation_image
rows = len(elevation_image)
cols = len(elevation_image[0])

elevation_image = (elevation_image - np.min(elevation_image))
curr_max_x = np.max(elevation_image)
curr_min_x = np.min(elevation_image)
        
elevation = transforms(elevation_image).to(torch.device("cuda"))
# after the totensor transform, the image range is within [0, 1]
# we now need to scale it to range [-1, 1] for DDPM 
max_x_new = 1.0
min_x_new = -1.0
elevation = scale_to_range(elevation, curr_min_x, curr_max_x, min_x_new, max_x_new)


# preparations 
mask_area_ratio = [0.2, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0]
# since the mask ratio does not account for the overlapping area, it is possible that the actual mask ratio is smaller than the target ratio 
masks = [np.array(generate_mask(rows, cols, mask_area_ratio[i], elevation)) for i in range(len(mask_area_ratio))]
true_mask_ratio = []
for m in masks:
    true_mask_ratio.append(m.sum()/(rows*cols))
masks = [-torch.from_numpy(mask).view(1, 1, rows, cols).
         repeat(num_samples, 1, 1, 1).
         to("cuda").float() 
         for mask in masks]


# choose 6 different forward steps 
ts = torch.tensor([875, 875, 875, 875, 875, 875, 875], device=torch.device("cuda"))
# every diffusion step is used to generate 5 samples to test the diversity 
num_samples = 1
repeated_original_data_sample = elevation.unsqueeze(dim=0).repeat(num_samples, 1, 1, 1)
# we add noises to the denoised sample
edited_samples = []
for i in range(ts.shape[0]):
    edited_x = editor(repeated_original_data_sample, ts[i:i+1].item(), keep_intermediate=False, mask=masks[i], lamb=0.9 - 0.1*i)
    edited_samples.append(edited_x[0].cpu().detach().numpy().reshape(rows, cols))

vis_samples = edited_samples


for id in range(len(vis_samples)):
    elevation_image = vis_samples[i]

    np.savetxt("./test_imgs/test/"+str(id)+".txt", elevation_image)