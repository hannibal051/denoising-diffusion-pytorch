from functools import partial 
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.var_scheduler import linear_beta_schedule 
from models.util import scale_to_range

class DenoisingDiffusionModel(nn.Module):
    def __init__(self, 
                 unet, 
                 total_diffusion_steps=1000, 
                 var_schedule='linear'):
        """A minimal implementation of the Denoising Diffusion Probabilistic Model (DDPM)
        """
        super().__init__()
        self.unet = unet
        self.total_diffusion_steps = total_diffusion_steps
        self.var_schedule = var_schedule
        if self.var_schedule == 'linear':
            self.betas = linear_beta_schedule() 
        else:
           NotImplementedError(f'Variance schedule {var_schedule} is not implemented.') 
        
        # variables for calculating the (1) forward process dist. params, (2) reverse process dist. params, and (3) loss 
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0) 
        self.alpha_bars_sqrt = torch.sqrt(self.alpha_bars) 
        self.one_minus_alpha_bars = 1 - self.alpha_bars

    def forward(self, x, t):
        """Predicts the noise of the reverse diffusion process ϵ(xₜ, t) based on the time and the current input.
        """
        predicted_noise = self.unet(x, t)
        return predicted_noise

    @torch.no_grad() 
    def q_mean_variance(self, x0, t):
        """Calculates the mean and the variance of the forward process at t, given the initial input 
        """
        b = x0.shape[0]
        # todo: for now, we assume an image input
        alpha_bars_sqrt = self.alpha_bars_sqrt[t].view(b, 1, 1, 1)
        one_minus_alpha_bars = self.one_minus_alpha_bars[t].view(b, 1, 1, 1)
        mean = alpha_bars_sqrt * x0
        var = one_minus_alpha_bars
        return mean, var

    @torch.no_grad() 
    def q_sample(self, x0, t, noise=None):
        """The forward diffusion process: q(x_{t}|x_0), where t = 0, ..., timestep

        Args:
            x (torch.tensor): The original input, shape (b, c, h, w) 
            timestep (torch.tensor): The total number of diffusion steps, shape (b, )
            noise (torch.tensor, optional): The noise to be added to the input. Defaults to None.
        Returns:
            A batch of noisy output at the corresponding timesteps, shape (b, c, h, w). 
        """
        if noise is None:
            noise = torch.randn_like(x0)
        mean, var = self.q_mean_variance(x0, t)
        return mean + torch.sqrt(var) * noise
    
    @torch.no_grad()
    def p_mean_variance(self, x, t):
        """Computes the mean and the variance of the reverse process p(x_{t-1}|x_t), where t = 1, ..., timestep
        """
        predicted_noise = self.forward(x, self._scale_time(t))
        b = x.shape[0]
        # todo: for now, we assume an image input
        # alpha_bars_sqrt = self.alpha_bars_sqrt[t].view(b, 1, 1, 1)
        alpha_sqrt = torch.sqrt(1 - self.betas[t]).view(b, 1, 1, 1)
        one_minus_alpha_bars_sqrt = torch.sqrt(self.one_minus_alpha_bars[t].view(b, 1, 1, 1))
        beta = self.betas[t].view(b, 1, 1, 1)
        # mean
        mu = 1 / alpha_sqrt * (x - beta*predicted_noise/one_minus_alpha_bars_sqrt)
        # todo: we currently choose the reverse process variance as beta_t
        var = beta
        return mu, var
    
    @torch.no_grad()
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
        
    @torch.no_grad()
    def p_sample(self, x, t, noise=None):
        """Performs the one-step reverse sampling process, i.e., x_{t-1} ~ p(x_{t-1}|x_t) given the current x and the timestep t.
        The sampled x is: xₜ₋₁ = μ(xₜ, t)+σ(xₜ, t) * ε 

        Args:
            x (torch.tensor): Noisy input, shape (b, c, h, w) 
            t (torch.tensor): Total number of steps to perform, shape (b, ) 
        """
        if noise is None:
            noise = torch.randn_like(x)
        mu, var = self.p_mean_variance(x, t)
        noise = noise * 0.0 if t[0].item() == 0 else noise
        # todo: reverse process variance is currently set to beta_t;
        # todo: but there are other schedules like the ones mentioned in the paper
        x_prev = mu + torch.sqrt(var) * noise
        # clip to make the input similar to the Gaussian distribution
        x_prev = torch.clamp(x_prev, -1, 1)
        return x_prev 

    @torch.no_grad()
    def p_sample_loop(self, x_T, T):
        """
        Iteratively sample from the given time step $T$ and the input $x_T$ to the first timestep $x_0$. 
        Store the intermediate samples in a list.
        Note that we assume each image example has the same diffusion steps T. 
        """
        p_samples = []
        p_sample = x_T.clone()
        b, c, h, w = p_sample.shape
        for timestep in reversed(range(T)):
            p_samples.append(p_sample)
            t = torch.tensor([timestep, ] * b).to(x_T.device)
            p_sample = self.p_sample(p_sample, t)
        p_samples.append(p_sample)
        p_samples = torch.stack(p_samples, dim=0)
        # scale the input from [-1, 1] to [0, 1]
        p_samples = scale_to_range(p_samples, -1, 1, 0, 1)
        return p_samples 
            
    def loss(self, x):
        """Given the original input x, we compute the DDPM simplified loss by 
        1. sample (b, ) timesteps from [0, total_diffusion_steps-1] 
        2. sample (b, c, h, w) noise from N(0, 1)
        3. compute the loss
        """
        b = x.shape[0]
        t = torch.randint(0, self.total_diffusion_steps, size=(b, )).to(x.device)
        noise = torch.randn_like(x)
        alpha_bars_sqrt = self.alpha_bars_sqrt[t].view(b, 1, 1, 1)
        one_minus_alpha_bars_sqrt = torch.sqrt(self.one_minus_alpha_bars[t]).view(b, 1, 1, 1)
        predicted_noise = self.forward(alpha_bars_sqrt*x+one_minus_alpha_bars_sqrt*noise, self._scale_time(t))
        # todo: in some advanced implementations, the loss is first averaged over the non-batch dimensions;
        # todo: then, the batch dimension loss is weighted based on the timestep t 
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def _scale_time(self, t):
        return t.float() / self.total_diffusion_steps
