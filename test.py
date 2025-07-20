import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pylab as plt
from lib.data.dataset import MRIDataset, transform
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader
import diffusers
from lib.utils.load_model import load_dict
from eval import evaluate
from lib.model.encoder import ImageEncoder
import torch
import swanlab
def test():
    swanlab.init(
    # 设置项目名
    project="test",
    
    # 设置超参数
    config={
        "learning_rate": 1e-4,
        "architecture": "unet",
        "dataset": "t1c",
        "epochs": 500
    }
    )
    eval_transform = transform()
    dataset = MRIDataset('./data', "total_list.txt", transforms=eval_transform)
    indices = range(dataset.__len__())
    eval_dataloader = DataLoader(dataset, batch_size=12, num_workers=4)
    unet = diffusers.UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
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
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    loaded_model = load_dict(unet, './ckpt/epoch_1.pth', optimizer=optimizer)
    model = loaded_model['model']
    noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(50)
    Imagencoder = ImageEncoder()
    evaluate(model=model, epoch=1, eval_dataloader=eval_dataloader, image_encoder=Imagencoder, noise_scheduler=noise_scheduler, swanlab=swanlab)

if __name__ == '__main__':
    test()