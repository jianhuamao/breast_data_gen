import torch
from lib.model.encoder import ImageEncoder
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from lib.eval.eval import evaluate
from diffusers import StableDiffusionInpaintPipeline
from lib.utils.load_model import load_dict
import time
from diffusers.models.unets.unet_2d_blocks import  DownBlock2D
def train_loop(config, model, noise_scheduler, optimizer, scheduler, train_dataloader, eval_dataloader, swanlab, device='cuda'):
    global_step = 0
    start_epoch = config.start_epoch
    image_encoder = ImageEncoder().to(device)
    conv_in = torch.nn.Conv2d(1, 128, kernel_size=3, padding=1).to(device)
    db = nn.ModuleList([])
    db_list = [128, 256, 512, 768, 1280]
    for i, num in enumerate(db_list):
        db.append(DownBlock2D(
        num_layers=1,
        in_channels=num,
        out_channels=db_list[i+1] if i < len(db_list)-1 else 1280,
        temb_channels=512))
    db.to(device)
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        model.train()
        print(config.isDebug) 
        train_time_0 = time.time()
        for data in train_dataloader:
            image = data['image'].to(device)
            gt = data['gt'].to(device)
            mask = data['mask'].to(device)
            mask = conv_in(mask)
            t_emb = model.get_time_embed(sample=mask, timestep=1000)
            emb = model.time_embedding(t_emb)
            for dblock in db:
                mask, _ = dblock(mask, emb)
            mask = mask.flatten(-2).transpose(1, 2)
            cond = image
            # Sample noise to add to the images
            noise = torch.randn_like(gt)
            bs = image.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=image.device).long()

            noisy_gt = noise_scheduler.add_noise(gt, noise, timesteps)
            noisy_gt = torch.cat([noisy_gt, cond], dim=1)
            # Predict the noise residual
            noise_pred = model(noisy_gt, timesteps, encoder_hidden_states=mask, return_dict=False)[0]
            loss = get_loss(noise_pred=noise_pred, noise=noise)
            loss.backward()
            swanlab.log({'loss': loss.item()})
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
        train_time_1 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        swanlab.log({
            'lr': current_lr,
            'train_time': train_time_1 - train_time_0 
                     })

        scheduler.step()

        should_eval = config.isDebug or ((epoch + 1) % 20 == 0 and epoch != 0)
        if should_eval:
            model.eval()
            # pipeline = diffusers.DDPMPipeline(unet=model.modules, scheduler=noise_scheduler)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_config': noise_scheduler.config,
                'epoch': epoch,
                'loss': loss.item()
            }, f'./ckpt/{config.name}_epoch_{epoch+1}.pth')
            eval_time_0 = time.time()
            evaluate(config, model, epoch+1, eval_dataloader, image_encoder, noise_scheduler, swanlab, device=device)
            eval_time_1 = time.time()
            swanlab.log({'eval_time': eval_time_1 - eval_time_0})
    
def get_loss(noise_pred, noise):
    '''
    mse loss
    '''
    criterition = torch.nn.MSELoss()
    loss = criterition(noise_pred, noise)
    return loss

