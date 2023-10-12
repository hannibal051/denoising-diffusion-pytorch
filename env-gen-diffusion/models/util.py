import torch

def check_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2 
    print("Model size: {:.2f} MB".format(size_all_mb))
    return sizｅ_all_mb

def scale_to_range(x, curr_min_x, curr_max_x, min_x_new, max_x_new):
    """
    This function is useful for scaling the input of the DDPM to the range of [-1, 1],
    and the output of the DDPM to the range of [0, 1] 
    """
    return (x - curr_min_x) * (max_x_new - min_x_new) / (curr_max_x - curr_min_x) + min_x_new

class ScaleToRange:
    def __init__(self, min_x, max_x, min_x_new: float = -1, max_x_new: float = 1):
        self.min_x = min_x 
        self.max_x = max_x
        self.min_x_new = min_x_new
        self.max_x_new = max_x_new

    def __call__(self, img_tensor):
        # Scale the image tensor to the target range
        # img_tensor = (img_tensor - min_pixel_value) * (self.max_value - self.min_value) / (max_pixel_value - min_pixel_value) + self.min_value
        return scale_to_range(img_tensor, self.min_x, self.max_x, self.min_x_new, self.max_x_new) 
