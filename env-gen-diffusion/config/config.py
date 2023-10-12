"""
The default configuration file for the DDPM model. 
"""
from yacs.config import CfgNode as CN

_C = CN()

"""
Parameters for the UNet Model 
"""
_C.unet = CN()
# the number of input channels
_C.unet.in_channels = 1
# the number of output channels
_C.unet.out_channels = 1
# the number of channels in the first convolutional layer
_C.unet.base_channel_count = 64
"""
DDPM parameters
"""
_C.ddpm = CN()
_C.ddpm.total_diffusion_steps = 1000 
_C.ddpm.var_schedule = "linear" 

"""
Training parameters
"""
_C.train = CN()
# the batch size for training
_C.train.batch_size = 256 
# the total number of epochs for training
_C.train.num_epochs = 100
# the learning rate for training
_C.train.lr = 1e-3
# the dataset to use for training
_C.train.dataset = "MINIST" # MINIST, CIFAR, ELEVATION
# the data root path 
_C.train.data_root = ""
# the save directory
_C.train.model_save_dir = "" 

"""
Validation parameters
"""
_C.eval = CN()
# the interval for validating and saving the model
_C.eval.val_epochs = 3 
# the number of images to sample for validation
_C.eval.num_samples = 16 # 4x4 grid of images
# the number of timesteps to sample for validation
_C.eval.num_diffusion_steps = 1000

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values"""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()