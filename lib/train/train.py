import torch
from lib.model.encoder import hidden_Encoder
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from lib.eval.eval import evaluate
from diffusers import StableDiffusionInpaintPipeline
from lib.utils.load_model import load_dict
import time
from lib.utils.save_model import save_model_hook
from lib.utils.text_encode import text_encode
import einops
def train_loop(config, model, noise_scheduler, optimizer, scheduler, train_dataloader, eval_dataloader, swanlab, device='cuda'):
    global_step = 0
    start_epoch = config.start_epoch
    if config.sd:
        hidden_states_encoder = hidden_Encoder().to(device) 
    else:
        hidden_states_encoder = None
    #train model 
    if isinstance(model, dict):
        unet = model['unet'].to(device)
        tokenizer = model['tokenizer']
        text_encoder = model['text_encoder'].to(device)
        vae = model['vae'].to(device)
        unet_lora_layers = model['unet_lora_layers'].to(device)
        uni_text_prompt = text_encode(tokenizer, 'a low quality image.', text_encoder).repeat(config.train_batch_size, 1, 1)
        part_text_pompt = text_encode(tokenizer, 'a high quality image.', text_encoder).repeat(config.train_batch_size, 1, 1)
        text_prompt = torch.cat([uni_text_prompt.unsqueeze(1), part_text_pompt.unsqueeze(1)], dim=1)
        text_prompt = einops.rearrange(text_prompt, 'b l c  h-> (b l) c h')
    #text_encode
    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        unet.train()
        train_time_0 = time.time()
        for data in train_dataloader:
            image = data['image'].to(device)
            gt = data['gt'].to(device)
            mask = data['mask'].to(device)
            if vae is not None:
                gt = vae.encode(gt).latent_dist.sample()
                image = vae.encode(image).latent_dist.sample()
                model_input = torch.cat([gt.unsqueeze(1), image.unsqueeze(1)], dim=1)
                model_input = einops.rearrange(model_input, 'b l c h w -> (b l) c h w')
            else:
                model_input = gt
            if hidden_states_encoder is not None:
                latent_mask = hidden_states_encoder(unet, mask)
            # Sample noise to add to the images
            noise = torch.randn_like(model_input)
            bs = model_input.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=image.device).long()
            noisy_gt = noise_scheduler.add_noise(model_input, noise, timesteps)
            # Predict the noise residual
            if text_prompt is not None :
                noise_pred = unet(noisy_gt, timesteps, encoder_hidden_states=text_prompt, return_dict=False)[0]
            else:
                noisy_gt = torch.cat([noisy_gt, image], dim=1)
                noise_pred = unet(noisy_gt, timesteps, return_dict=False)[0]
            loss = get_loss(noise_pred=noise_pred, noise=noise, mask=None)
            loss.backward()
            swanlab.log({'loss': loss.item()})
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
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
            unet.eval()
            # pipeline = diffusers.DDPMPipeline(unet=model.modules, scheduler=noise_scheduler)
            if config.model_name == 'stabel_diffusion':
                torch.save({
                    'unet_lora_layers': unet_lora_layers.state_dict()
                    }, f'./ckpt/lora/{config.name}_epoch_{epoch+1}.pth')
            else:
                torch.save({
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_config': noise_scheduler.config,
                'epoch': epoch,
                'loss': loss.item()
                    }, f'./ckpt/{config.name}_epoch_{epoch+1}.pth')
            # save_model_hook(models=[model], unet_lora_layers=[], weights=[1.0], output_dir=f'./ckpt/lora/{config.name}')
            eval_time_0 = time.time()
            model_and_text = {
                'unet': unet,
                'text_prompt': text_prompt,
                'vae': vae
            }
            evaluate(config, model_and_text, epoch+1, eval_dataloader, hidden_states_encoder, noise_scheduler, swanlab, device=device)
            eval_time_1 = time.time()
            swanlab.log({'eval_time': eval_time_1 - eval_time_0})
    
def get_loss(noise_pred, noise, mask=None):
    '''
    mse loss
    '''
    criterition = torch.nn.MSELoss()
    loss_uni = criterition(noise_pred, noise)
    if mask is not None:
        par_noise_pred = noise_pred * mask
        par_noise = noise * mask 
        loss_par = criterition(par_noise_pred, par_noise)
        loss = loss_uni + 0.5*loss_par
    else:
        loss = loss_uni
    return loss

