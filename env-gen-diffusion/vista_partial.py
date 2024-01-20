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

def generate_mask_by_gradient(H, W, mask_area_ratio, map_txt):
    mask = np.zeros_like(map_txt[0])
    # mask = map_txt[0].copy()

    target_area = H * W * mask_area_ratio

    gradient = np.abs(np.gradient(map_txt[0]))
    gradient = np.sqrt(np.square(gradient[0]) + np.square(gradient[1]))

    min_grad_indices = np.dstack(np.unravel_index(np.argsort(gradient.ravel()), gradient.shape))[0]
    grad_filter = set()
    r = 10
    center_circle = True
    for id in range(H * W):
        xx = min_grad_indices[id][0]
        yy = min_grad_indices[id][1]
        if center_circle:
            xx = H // 2
            yy = W // 2
            r = 30
        for rr in range(-r+1, r):
            y1 = yy + rr
            for r1 in range(abs(rr)-r+1, r-abs(rr)):
                x1 = xx + r1
                if x1>=0 and x1<H and y1>=0 and y1<W:
                    # print(len(grad_filter), target_area)
                    if len(grad_filter) >= target_area:
                        break
                    
                    grad_filter.add((x1, y1, 1))
                    # if map_txt[0][x1][y1] >= 0:
                    #     grad_filter.add((x1, y1, 1))
                    # else:
                    #     grad_filter.add((x1, y1, -1))
        if center_circle:
            break
    for id in grad_filter:
        mask[id[0]][id[1]] = id[2]
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
# terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/output_hh.tif"

terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/elevation/test_map4.txt"
# terrain_path = "/Users/sp/sp/sim-to-real-offroad/assets/tif/output_hh.tif"

if terrain_path[-3:] == 'tif':
    elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)[:128*4, :128*4]
elif terrain_path[-3:] == 'txt':
    elevation_image = np.loadtxt(terrain_path, dtype=np.float32, delimiter=',')

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
# masks = [np.array(generate_mask(rows, cols, mask_area_ratio[i], elevation)) for i in range(len(mask_area_ratio))]
masks = [np.array(generate_mask_by_gradient(rows, cols, mask_area_ratio[i], elevation.cpu().numpy().copy())) for i in range(len(mask_area_ratio))]

masks = [torch.from_numpy(mask).view(1, 1, rows, cols).
         repeat(num_samples, 1, 1, 1).
         to("cuda").float() 
         for mask in masks]

# choose 6 different forward steps 
# ts = torch.tensor([125, 125, 125, 125, 125, 125, 125], device=torch.device("cuda"))
ts = torch.tensor([250], device=torch.device("cuda"))

# every diffusion step is used to generate 5 samples to test the diversity 
num_samples = 1
repeated_original_data_sample = elevation.unsqueeze(dim=0).repeat(num_samples, 1, 1, 1)
# we add noises to the denoised sample
edited_samples = []
for i in range(ts.shape[0]):
    edited_x = editor(repeated_original_data_sample, ts[i:i+1].item(), keep_intermediate=False, mask=masks[i], lamb=0.1)
    edited_samples.append(edited_x[0].cpu().detach().numpy().reshape(rows, cols))

# for i in range(len(edited_samples)):
#     mask_i = masks[i].cpu().numpy()[0][0]
#     edited_samples[i] = np.multiply(elevation.cpu().numpy()[0], 1 - mask_i) + np.multiply(edited_samples[i], mask_i)

for id in range(len(edited_samples)):
    np.savetxt("./test_imgs/test/"+str(id)+".txt", edited_samples[id])