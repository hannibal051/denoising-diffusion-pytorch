import os
import sys
# Add the root directory to the sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
import unittest
import os
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from models.diffusion import DenoisingDiffusionModel 

class TestDenoisingDiffusionModel(unittest.TestCase):
    def setUp(self) -> None:
        image_root_paths = "/media/hdd/env-gen-diffusion/elevation" 
        images = [imageio.imread(os.path.join(image_root_paths, f"{i}.png")) for i in range(4)] 
        # make it to gray scale
        images = np.array([np.dot(images[i][..., :3], [0.2989, 0.5870, 0.1140]) for i in range(4)])
        self.images = torch.from_numpy(images).to('cuda').float()/255.
        self.images = self.images.view(4, 1, 128, 128)
        self.ddpm = DenoisingDiffusionModel(None, 'linear')

    def test_q_sample(self):
        # we select four timesteps (t, b)
        t = torch.tensor([[0, 0, 0, 0], 
                          [1, 1, 1, 1],
                          [10, 10, 10, 10],
                          [50, 50, 50, 50], 
                          [100, 100, 100, 100],
                          [999, 999, 999, 999]], device='cuda').long()
        for i in range(t.shape[0]):
            samples = self.ddpm.q_sample(self.images, t[i])
            samples = samples.view(4, 128, 128)
            fig, axes = plt.subplots(2, 2)
            for j in range(2):
                for k in range(2):
                    axes[j, k].imshow(samples[j*2+k].cpu().numpy())
            fig.suptitle(f"t={t[i, 0].item()}")
            plt.savefig(f"tests/test_media/q_sample_{t[i, 0].item()}.png", bbox_inches='tight', dpi=200)
            plt.show()

    def test_reverse(self):
        pass

if __name__ == '__main__':
    unittest.main()