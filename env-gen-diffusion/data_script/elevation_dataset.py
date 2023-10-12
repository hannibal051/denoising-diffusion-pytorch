from glob2 import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from models.util import scale_to_range
import torchvision.transforms as transforms

class ElevationDataset(Dataset):
    def __init__(self, data_dir, transforms=transforms.ToTensor()):
        super().__init__()
        elevation_files = glob(data_dir + "/*.png")
        if len(elevation_files):
            self.elevation_data = []
            for elevation_file in elevation_files:
                elevation_img = Image.open(elevation_file).convert("L")
                self.elevation_data.append(np.array(elevation_img))
            self.transforms = transforms
            self.elevation_data = np.array(self.elevation_data)
        else:
            elevation_files = glob(data_dir + "/*.txt")
            self.elevation_data = []
            for elevation_file in elevation_files:
                elevation_img = np.loadtxt(elevation_file, delimiter=",")
                self.elevation_data.append(elevation_img)
            self.transforms = transforms
            self.elevation_data = np.array(self.elevation_data, dtype=np.float32)

    def __len__(self):
        return self.elevation_data.shape[0]
    
    def __getitem__(self, index):
        elevation = self.elevation_data[index]
        curr_max_x = np.max(elevation)
        curr_min_x = np.min(elevation)
        elevation = self.transforms(elevation)
        # after the totensor transform, the image range is within [0, 1]
        # we now need to scale it to range [-1, 1] for DDPM 
        max_x_new = 1.0
        min_x_new = -1.0
        elevation = scale_to_range(elevation, curr_min_x, curr_max_x, min_x_new, max_x_new)
        # todo: we add a dummy tensor for the label, but we don't use it in the model
        return elevation, (max_x_new - min_x_new) / (curr_max_x - curr_min_x)