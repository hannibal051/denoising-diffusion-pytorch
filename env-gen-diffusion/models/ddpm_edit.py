import torch
from tqdm import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
# from config.config import get_cfg_defaults

class Editor():
    def __init__(self, model_path, num_iterations=1) -> None:
        """Use DDPM to edit the input images. This class includes both global and partial editing.
        """
        self.model = Unet(
            dim = 16,
            channels = 1,
            dim_mults = (1, 2, 4)
        ).cuda()
        # in_channels out_channels base_channel_count

        self.ddpm = GaussianDiffusion(
            self.model,
            image_size = 128,
            timesteps = 1000,           # number of steps, default 1000
            sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ).cuda()

        self.ddpm.load_state_dict(torch.load(model_path)['model'])#, strict=False)
        self.num_iterations = num_iterations
    
    @torch.no_grad()
    def p_sample_with_noise(self, x, t, noise):
        """Sample from the DDPM reverse process with a given noise. 
        """
        # copied from the original ddpm code
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.ddpm.p_mean_variance(x = x, t = batched_times, x_self_cond = None, clip_denoised = True)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.no_grad()
    def __call__(self, x: torch.Tensor, total_timestep: int, mask: torch.Tensor=None, keep_intermediate=False, lamb=0.5) -> torch.Tensor:
        """Edit the original image x from an initial time t. 
        If mask (0 for non-editable, 1 for editable) is given, only edit the unmasked area.
        The output is the edited image normalized to [0, 1], and the input should be normalized to [-1, 1]. 
        """
        shape = x.shape
        # 1. noise the image till T timesteps
        timesteps = torch.full((shape[0],), total_timestep-1, dtype=torch.long, device=x.device)
        noisy_x = self.ddpm.q_sample(x, timesteps)
        print(torch.var_mean(noisy_x))
        if keep_intermediate:
            intermediate_xs = [noisy_x]
        if mask is None:
            # 2. edit the image by denoising
            # todo: we can choose whether we want to use ddim to make the sampling faster or the original ddpm
            edited_x = noisy_x.clone()
            for t in tqdm(reversed(range(0, total_timestep)), desc = 'sampling loop time step', total = total_timestep):
                edited_x, _ = self.ddpm.p_sample(edited_x, t, None)
                if keep_intermediate:
                    intermediate_xs.append(edited_x)
            # edited_x = unnormalize_to_zero_to_one(edited_x)
        else:
            return self.ddpm.interpolate(x, mask, t = total_timestep, lam = lamb)

            edited_x = noisy_x.clone()
            # 2. partial reverse process
            for t in tqdm(reversed(range(0, total_timestep)), desc = 'sampling loop time step', total = total_timestep):
                # we sample from the current noisy image to keep the noise level the same as the edited part
                noise = torch.randn_like(x)
                edited_x, _ = self.p_sample_with_noise(edited_x, t, noise)
                # edited_x, _ = self.ddpm.p_sample(edited_x, t, None)
                
                if t > 0:
                    timesteps = torch.full((shape[0],), t-1, dtype=torch.long, device=x.device)
                    non_edited_x = self.ddpm.q_sample(x, timesteps)
                else:
                    non_edited_x = x 
                edited_x = non_edited_x * (1 - mask) + edited_x * mask
                if keep_intermediate:
                    intermediate_xs.append(edited_x)
        if keep_intermediate:
            # intermediate_xs is of shape (batch_size, t, C, H, W)
            return unnormalize_to_zero_to_one(edited_x), unnormalize_to_zero_to_one(torch.stack(intermediate_xs, dim=1))
        else:
            return unnormalize_to_zero_to_one(edited_x)
    
    def sample(self, shape):
        # a wrapper for ddpm.p_sample_loop
        return self.ddpm.p_sample_loop(shape)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    editor = Editor("/home/sp/terrain/denoising-diffusion-pytorch/results/model-3.pt")
    # random sample one image
    shape = (1, 1, 128, 128)
    sample = editor.sample(shape)

    # edited_sample, intermediate = editor(sample, 500, keep_intermediate=True)

    # fig, axis = plt.subplots(1, 2)
    # axis[0].imshow(sample[0, 0].cpu().detach().numpy())
    # axis[1].imshow(edited_sample[0, 0].cpu().detach().numpy())
    # print(intermediate.shape)
    # plt.show()

    mask = torch.zeros(shape, device=sample.device)
    mask[:, :, :64] = 1
    edited_sample, intermediate = editor(sample, 50, mask=mask, keep_intermediate=True)
    print((edited_sample < 0).sum())
    print((intermediate[:, -1] < 0).sum())
    fig, axis = plt.subplots(1, 3)
    sample_np = sample[0, 0].cpu().detach().numpy()
    edited_sample_np = edited_sample[0, 0].cpu().detach().numpy()
    diff = (sample_np - edited_sample_np)
    vmin = min(sample_np.min(), edited_sample_np.min(), diff.min())
    vmax = max(sample_np.max(), edited_sample_np.max(), diff.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    axis[0].imshow(sample_np, norm=norm)
    axis[1].imshow(edited_sample_np, norm=norm)
    axis[2].imshow(diff, norm=norm)
    plt.show()
    