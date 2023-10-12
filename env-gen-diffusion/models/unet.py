import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""The UNet Parts"""
class ResidualBlock(nn.Module):
    """(convolution => [BN] => SiLU) * 2"""

    def __init__(self, in_channels, out_channels, embed_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # time embedding layer is added to the middle layer, after the BN1
        self.time_embed_layer = nn.Sequential(
            nn.Linear(embed_dim, mid_channels),
            nn.SiLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(mid_channels//2, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(out_channels//2, out_channels)
        if in_channels != out_channels:
            # linear projection shortcut, see https://arxiv.org/pdf/1512.03385.pdf section shortcut connections 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(out_channels//2, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_embed):
        residual = x
        # first layer
        out = self.conv1(x)
        out = self.group_norm1(out)
        out = F.silu(out)

        # time embedding layer
        t_embed = self.time_embed_layer(t_embed)
        while len(t_embed.shape) < len(x.shape):
            t_embed = t_embed[..., None]
        out = out + t_embed

        # second layer
        out = self.conv2(out)
        out = self.group_norm2(out)

        # residual connection
        residual = self.shortcut(residual)

        out = out + residual
        out = F.silu(out)
        return out 


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, t_embed_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ResidualBlock(in_channels, out_channels, t_embed_dim)
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     DoubleConv(in_channels, out_channels, t_embed_dim)
        # )

    def forward(self, x, t_embed):
        x = self.maxpool(x)
        x = self.conv(x, t_embed)
        return x 


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, t_embed_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels, t_embed_dim)

    def forward(self, x1, x2, t_embed):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_embed)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_channels, n_out, base_channel_count=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_out = n_out
        # the time embedding layer is the same as the one used in the improved diffusion model 
        # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/unet.py#L476
        self.base_channel_count = base_channel_count 
        self.t_embed_dim = self.base_channel_count * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channel_count, self.t_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.t_embed_dim, self.t_embed_dim)
        )
        self.inc = (ResidualBlock(n_channels, 
                               self.base_channel_count, 
                               self.t_embed_dim))
        self.down1 = (Down(self.base_channel_count, 
                           self.base_channel_count*2, 
                           self.t_embed_dim))
        self.down2 = (Down(self.base_channel_count*2, 
                           self.base_channel_count*4, 
                           self.t_embed_dim))
        self.down3 = (Down(self.base_channel_count*4, 
                           self.base_channel_count*8, 
                           self.t_embed_dim))
        self.down4 = (Down(self.base_channel_count*8, 
                           self.base_channel_count*16, 
                           self.t_embed_dim))
        self.up1 = (Up(self.base_channel_count*16, 
                       self.base_channel_count*8, 
                       self.t_embed_dim))
        self.up2 = (Up(self.base_channel_count*8, 
                       self.base_channel_count*4, 
                       self.t_embed_dim))
        self.up3 = (Up(self.base_channel_count*4, 
                       self.base_channel_count*2, 
                       self.t_embed_dim))
        self.up4 = (Up(self.base_channel_count*2, 
                       self.base_channel_count, 
                       self.t_embed_dim))
        self.outc = (OutConv(self.base_channel_count, 
                             n_out))

    def forward(self, x, t):
        t_embed = self.time_embed(timestep_embedding(t, self.base_channel_count)) 
        x1 = self.inc(x, t_embed)
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x4 = self.down3(x3, t_embed)
        x5 = self.down4(x4, t_embed)
        x = self.up1(x5, x4, t_embed)
        x = self.up2(x, x3, t_embed)
        x = self.up3(x, x2, t_embed)
        x = self.up4(x, x1, t_embed)
        out = self.outc(x)
        return out 

def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.
	
	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	half = dim // 2
	freqs = torch.exp(
		-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
	).to(device=timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	return embedding