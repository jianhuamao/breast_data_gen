import torch
from lib.model.encoder import ImageEncoder
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from eval import evaluate
from diffusers import StableDiffusionInpaintPipeline
from lib.utils.load_model import load_dict

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, device='cuda'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    criterition = torch.nn.MSELoss()
    global_step = 0
    device = 'cuda:0'
    # Now you train the model
    start_epoch = config.start_epoch
    image_encoder = ImageEncoder().to(device)
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        model.train()

        for data in train_dataloader:
            image = data['image'].to(device)
            gt = data['gt'].to(device)
            # mask = data['mask'].to(device)
            cond = image
            # encoded_image = image_encoder(image)
            # Sample noise to add to the images
            noise = torch.randn_like(gt)
            bs = image.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=image.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_gt = noise_scheduler.add_noise(gt, noise, timesteps)
            noisy_gt = torch.cat([noisy_gt, cond], dim=1)
            # Predict the noise residual
            noise_pred = model(noisy_gt, timesteps, return_dict=False)[0]
            loss = criterition(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
        # if epoch == 0:
        if (epoch+1) % 50 == 0 and epoch != 0:
            model.eval()
            # pipeline = diffusers.DDPMPipeline(unet=model.modules, scheduler=noise_scheduler)
            evaluate(model, epoch+1, eval_dataloader, image_encoder, noise_scheduler)
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_config': noise_scheduler.config,
                    'epoch': epoch,
                    'loss': loss.item()
                }, f'./ckp/epoch_{epoch}.pth')