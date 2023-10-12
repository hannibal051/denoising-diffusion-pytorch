import torch

def linear_beta_schedule(min_beta=1e-4, max_beta=0.02, num_timesteps=1000, device='cuda'):
    """Linearly increase the beta values from min_beta to max_beta within num_timesteps.
    """
    scale = 1000 / num_timesteps 
    beta_start = scale * min_beta 
    beta_end = scale * max_beta
    return torch.linspace(beta_start, beta_end, num_timesteps, device=device)

def cosine_beta_schedule():
    pass