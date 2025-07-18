import argparse
import os
from lib.config import trainingConfig
from lib.data.dataset import MRIDataset
import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import diffusers
from lib.model.encoder import ImageEncoder
from lib.model.UnetWithSigmoid import UNetWithSigmoid
import matplotlib.pyplot as plt
from train import train_loop
from lib.data.train_sampler import trainSampler
from log.log import train_log
from lib.utils.load_model import load_dict
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--train_model', type=str, default='single')
    parser.add_argument('--nproc_per_node',type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_sampler', type=int , default= 1100)
    parser.add_argument('--num_train_timesteps', type=int, default= 1000)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--pretrain_model_path', type=str, default= None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #GPU
    device = torch.device("cuda")
    #loader config
    config = trainingConfig(
        image_size=args.image_size,
        data_folder=args.data_folder,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        nproc_per_node=args.nproc_per_node,
        in_channels=args.in_channels,
        num_sampler = args.num_sampler,
        num_train_timesteps = args.num_train_timesteps,
        start_epoch = args.start_epoch,
        pretrain_model_path = args.pretrain_model_path
        )
    #load data
    train_dataset = MRIDataset(config.data_folder, "train_list.txt")
    eval_dataset = MRIDataset(config.data_folder, 'eval_list.txt')
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=5000)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=4, sampler=train_sampler)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, num_workers=4)
    unet = diffusers.UNet2DModel(
        sample_size=64,  # the target image resolution
        in_channels=6,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )
    model = unet
    model.to(device)
    if config.start_epoch > 0 and config.pretrain_model_path is not None:
        model = load_dict(model=model, pretrain_model_path=config.pretrain_model_path)
    noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader)

        
if __name__ == "__main__":
    main()
