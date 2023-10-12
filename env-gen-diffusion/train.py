from config.config import get_cfg_defaults
import torch
import torchvision
from torchvision.utils import make_grid
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models.unet import UNet
from models.diffusion import DenoisingDiffusionModel
from models.util import ScaleToRange
import matplotlib.pyplot as plt
import argparse
import torch.utils.tensorboard as tb
from data_script.elevation_dataset import ElevationDataset


def eval(model, last_images, num_diffusion_steps, num_samples, writer, global_step):
    """
    We evaluate the models in two ways:
    1. We add noise to the last image and see how well the model denoises it 
    2. We sample new images from the model to qualitatively evaluate the model
    """
    model.eval()
    last_images = last_images[:num_samples]
    # adding noise
    ts = torch.tensor([num_diffusion_steps-1, ] * num_samples).to('cuda')
    denoised_original_img = model.q_sample(last_images, ts)
    with torch.no_grad():
        # 1. denoise the noised last images
        denoised_original_img = model.p_sample_loop(denoised_original_img, num_diffusion_steps-1)
        # 2. sample new images
        new_sampled_img = model.p_sample_loop(torch.randn_like(last_images), num_diffusion_steps-1)
    original_grid = make_grid(last_images) 
    denoised_original_grid = make_grid(denoised_original_img[-1])
    new_sampled_grid = make_grid(new_sampled_img[-1])
    # store the de-noised images the tensorboard
    writer.add_image("original", original_grid, global_step=global_step)
    writer.add_image("denoised", denoised_original_grid, global_step=global_step)
    writer.add_image("new samples", new_sampled_grid, global_step=global_step)
    # let's try using matplotlib to plot the images
    fig, ax = plt.subplots(2, 8)
    for row in range(2):
        for col in range(8):
            ax[row, col].imshow(new_sampled_img[-1][row*8+col].squeeze().cpu().numpy(), cmap='gray')
    fig.savefig("test_imgs/denoised.png", dpi=200, bbox_inches='tight')
    plt.close()

def train(cfg):
    writer = tb.SummaryWriter(cfg.train.model_save_dir)
    unet = UNet(cfg.unet.in_channels, cfg.unet.out_channels, cfg.unet.base_channel_count).to('cuda')
    ddpm = DenoisingDiffusionModel(unet, 
                                   total_diffusion_steps=cfg.ddpm.total_diffusion_steps, 
                                   var_schedule=cfg.ddpm.var_schedule).to('cuda')
    optimizer = AdamW(ddpm.parameters(), lr=cfg.train.lr)
    if cfg.train.dataset == "MINIST":
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     ScaleToRange(0, 1)]) 
        dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=True,
                                             download=True, 
                                             transform=transforms) 
    elif cfg.train.dataset == "elevation":
        dataset = ElevationDataset(cfg.train.data_root)
    batch_size = cfg.train.batch_size 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    total_num_training_steps = cfg.train.num_epochs

    global_step = 0
    for e in range(total_num_training_steps):
        # todo: minist has labels, but our custom dataset does not; need a way to handle this 
        for i, (image, _) in enumerate(data_loader):
            ddpm.train()
            image = image.to('cuda')
            loss = ddpm.loss(image)
            optimizer.zero_grad()
            loss.backward()
            writer.add_scalar("loss", loss.item(), global_step=global_step)
            print("At epoch {}, iteration {}/{}, loss is {}".format(e, i, len(data_loader), loss.item()), end='\r')
            # gradient clipping (?)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        if e % cfg.eval.val_epochs == 0:
            with torch.no_grad():
                eval(ddpm, image, cfg.eval.num_diffusion_steps, cfg.eval.num_samples, writer, global_step)
                torch.save(ddpm.state_dict(), cfg.train.model_save_dir + "/models/model_{}.pt".format(e))


if __name__ == '__main__':
    config = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='Train a DDPM model')
    parser.add_argument('--train-cfg', type=str, default='minist', help='the configuration file for training')
    config.merge_from_file("config/{}.yaml".format(parser.parse_args().train_cfg))
    print("📑 Training Configuration 📑")
    print(config)
    print("⏩⏩⏩⏩Start Training⏩⏩⏩⏩")
    # start training
    train(config)
    