import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from log.log import train_log
from PIL import Image
import cv2 as cv
def path_gen(path_list:list, add_name):
     return [os.path.join(dir_name, add_name) for dir_name in path_list]
def is_path_exists(path_list:list):
     for path in path_list:
          if not os.path.exists(path):
               os.makedirs(path)
def evaluate(model, epoch, eval_dataloader, image_encoder, noise_scheduler, swanlab=None):
    noise_scheduler.set_timesteps(50) 
    device = torch.device("cuda:0")
    model = model.to(device)
    image_encoder = image_encoder.to(device).eval()
    folder = "./output"
    path_list = ['generate', 'image', 'gt', 'mask']
    path_list = [os.path.join(folder, name) for name in path_list]
    all_path = path_gen(path_list=path_list, add_name=f'epoch{epoch}')
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # init log
    mse_log = train_log('mse')
    mae_log = train_log('mae')
    psnr_log = train_log('psnr')
    ssim_log = train_log('ssim')
    num_idx = 1
    num_loop = 150
    with torch.no_grad():
        for data in eval_dataloader:
            image = data['image'].to(device) 
            gt = data['gt'].to(device)
            cond = image
            latent = torch.rand_like(gt).to(device)
            for t in noise_scheduler.timesteps:
                t = t.to(device)
                print(t.item())
                # generate images 
                x_in = torch.cat([latent, cond], dim=1)
                pre_noise = model(x_in, t.expand(latent.shape[0]), return_dict=False)[0] #if in_put = [gt, image], model(latent, ...) latent change to x_in
                latent = noise_scheduler.step(pre_noise, t, latent).prev_sample  
            # generated_image = torch.clamp(latent, 0, 1)
            generated_image = latent
            generated_image = torch.clamp(generated_image, -1, 1) * 0.5 + 0.5
            # convert to numpy
            generated_np = generated_image.detach().cpu().squeeze().numpy()
            ground_truth_np = gt.detach().cpu().squeeze().numpy()
            image_np = image.detach().cpu().squeeze().numpy()
            mask_np = data['mask'].detach().cpu().squeeze().numpy()
            bs = generated_np.shape[0]
            # generated_np = np.clip((generated_np + 1) / 2, 0, 1)
            if swanlab is not None:
                gen_list, image_list, gt_list, mask_list = [],[],[],[]
            while num_idx <= num_loop:
                for i in range(bs):
                    sub_path = path_gen(all_path, str(num_idx))
                    is_path_exists(sub_path)
                    gen_path, img_path, gt_path, mask_path = path_gen(sub_path, f'{i}.png')  
                    plt.imsave(gen_path, generated_np[i], cmap = 'gray')
                    plt.imsave(gt_path, ground_truth_np[i], cmap = 'gray')
                    plt.imsave(img_path, image_np[i], cmap='gray')
                    plt.imsave(mask_path, mask_np[i], cmap='gray')
                    if swanlab is not None:
                        gen_list.append(swanlab.Image(gen_path))
                        image_list.append(swanlab.Image(img_path))
                        gt_list.append(swanlab.Image(gt_path))
                        mask_list.append(swanlab.Image(mask_path))
                    num_idx += 1
                    if num_idx > num_loop:
                        break
            
        # 计算指标
        mse = np.mean((generated_np - ground_truth_np) ** 2)
        mae = np.mean(np.abs(generated_np - ground_truth_np))
        psnr_val = psnr(ground_truth_np, generated_np, data_range=1.0)
        ssim_val = ssim_fn(generated_image, gt).item()
        # 保存结果
        mse_list.append(mse)
        mae_list.append(mae)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    # 打印平均指标
    out_mse = f"epoch {epoch}, MSE: {np.mean(mse_list):.4f}"
    out_mae = f"epoch {epoch},MAE: {np.mean(mae_list):.4f}"
    out_psnr = f"epoch {epoch},PSNR: {np.mean(psnr_list):.4f}"
    out_ssim = f"epoch {epoch},SSIM: {np.mean(ssim_list):.4f}"
    print(out_mse)
    print(out_mae)
    print(out_psnr)
    print(out_ssim)
    mse_log.write_log(out_mse)
    mae_log.write_log(out_mae)
    psnr_log.write_log(out_psnr)
    ssim_log.write_log(out_ssim)
    if swanlab is not None:
        swanlab.log({'mse': np.mean(mse_list), 'mae': np.mean(mae_list), 'psnr': np.mean(psnr_list), 'ssim': np.mean(ssim_list)})
        swanlab.log({
            'generate': gen_list,
            'gt': gt_list,
            'image': image_list,
            'mask': mask_list
        })

