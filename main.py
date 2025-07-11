import argparse
import os
from lib.config import trainingConfig
from lib.data.dataset import MRIDataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import diffusers
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from lib.model.encoder import ImageEncoder
from lib.model.UnetWithSigmoid import UNetWithSigmoid
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--train_model', type=str, default='single')
    parser.add_argument('--nproc_per_node',type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #GPU
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    #loader config
    config = trainingConfig(
        image_size=args.image_size,
        data_folder=args.data_folder,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        nproc_per_node=args.nproc_per_node,
        in_channels=args.in_channels)
    #load data
    dataset = MRIDataset(config.data_folder, "total_list.txt")
    config.dataset = dataset
    indices = range(dataset.__len__())
    train_idx, eva_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eva_idx)
    train_dataloader = DataLoader(dataset, batch_size=16, num_workers=0, sampler=train_sampler)
    eval_dataloader = DataLoader(dataset, batch_size=16, num_workers=0, sampler=eval_sampler)
    unet = diffusers.UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=config.in_channels,  # the number of input channels, 3 for RGB images
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
    model = UNetWithSigmoid(unet)
    model.to(device)
    noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader)
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, device='cpu'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    # Now you train the model
    start_epoch = 0
    image_encoder = ImageEncoder()

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for image, label in train_dataloader:
            image = image.to(device)
            label = label.to(device)
            image_encoded = image_encoder(image)
            # Sample noise to add to the images
            noise = torch.randn(image.shape).to(image.device)
            bs = image.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=image.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_label = noise_scheduler.add_noise(label, noise, timesteps)
            # Predict the noise residual
            noise_pred = model(noisy_label, timesteps, encoder_hidden_states=image_encoded, return_dict=False)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
        model.eval()
        # pipeline = diffusers.DDPMPipeline(unet=model.modules, scheduler=noise_scheduler)
        evaluate(model, epoch, eval_dataloader, image_encoder, noise_scheduler)
        
def evaluate(model, epoch, eval_dataloader, image_encoder, noise_scheduler):
    out_output_folder = "./output"
    generate_path = os.path.join(out_output_folder, f'epoch_{epoch}')
    gt_path = os.path.join(out_output_folder, f'epoch_{epoch}')
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    device = 'cpu' 
    with torch.no_grad():
        for image, ground_truth in eval_dataloader: 
            shape = image.shape
            noise = torch.randn(shape).to(device)
            for t in noise_scheduler.timesteps:
                image = image.to(device)
                image_feature = image_encoder(image)
                ground_truth = ground_truth.to(device)
                bs = image.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=image.device).long()
                #generate noise
                
                # generate images 
                generated_image = model(noise, t.expand(noise.shape[0]), encoder_hidden_states=image_feature, return_dict=False)
                noise = noise_scheduler.step(generated_image, t, noise).prev_sample  
            generated_image = torch.clamp(generated_image, 0, 1)
            # convert to numpy
            generated_np = generated_image.squeeze().numpy()
            ground_truth_np = ground_truth.squeeze().numpy()
            if not os.path.exists(generate_path):
                os.makedirs(generate_path)
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            plt.imsave(os.path.join(generate_path, 'generate.png'), np.transpose(generated_np[0],(1, 2, 0)))
            plt.imsave(os.path.join(gt_path, 'gt.png'), np.transpose(ground_truth_np[0],(1, 2, 0)))

            # 计算指标
            mse = np.mean((generated_np - ground_truth_np) ** 2)
            mae = np.mean(np.abs(generated_np - ground_truth_np))
            psnr_val = psnr(ground_truth_np, generated_np, data_range=1.0)
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
            ssim_val = ssim_fn(generated_image, ground_truth).item()
            # 保存结果
            mse_list.append(mse)
            mae_list.append(mae)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    # 打印平均指标
    print(f"epoch {epoch}, MSE: {np.mean(mse_list):.4f}")
    print(f"epoch {epoch},MAE: {np.mean(mae_list):.4f}")
    print(f"epoch {epoch},PSNR: {np.mean(psnr_list):.4f}")
    print(f"epoch {epoch},SSIM: {np.mean(ssim_list):.4f}")
if __name__ == "__main__":
    main()
