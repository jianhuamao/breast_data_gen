import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from log.log import train_log
def evaluate(model, epoch, eval_dataloader, image_encoder, noise_scheduler):
    noise_scheduler.set_timesteps(50) 
    device = torch.device("cuda:0")
    model = model.to(device)
    image_encoder = image_encoder.to(device).eval()
    out_output_folder = "./output"
    generate_path = os.path.join(out_output_folder, f'epoch_{epoch}')
    gt_path = os.path.join(out_output_folder, f'epoch_{epoch}')
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
    with torch.no_grad():
        for data in eval_dataloader:
            image = data['image'].to(device) 
            gt = data['gt'].to(device)
            # mask = torch.rand_like(data['mask']).to(device)
            cond = image
            latent = torch.rand_like(gt).to(device)
            for t in noise_scheduler.timesteps:
                t = t.to(device)
                print(t.item())
                # generate images 
                x_in = torch.cat([latent, cond], dim=1)
                pre_noise = model(x_in, t.expand(latent.shape[0]), return_dict=False)[0]
                latent = noise_scheduler.step(pre_noise, t, latent).prev_sample  
            # generated_image = torch.clamp(latent, 0, 1)
            generated_image = latent
            generated_image = torch.clamp(generated_image, -1, 1) * 0.5 + 0.5
            # convert to numpy
            generated_np = generated_image.detach().cpu().squeeze().numpy()
            ground_truth_np = gt.detach().cpu().squeeze().numpy()
            image_np = image.detach().cpu().squeeze().numpy()
            mask_np = data['mask'].detach().cpu().squeeze().numpy()
            # generated_np = np.clip((generated_np + 1) / 2, 0, 1)
            if not os.path.exists(generate_path):
                os.makedirs(generate_path)
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            plt.imsave(os.path.join(generate_path, 'generate.png'), np.transpose(generated_np[0],(1, 2, 0)))
            plt.imsave(os.path.join(gt_path, 'gt.png'), np.transpose(ground_truth_np[0],(1, 2, 0)))
            plt.imsave(os.path.join(gt_path, 'image.png'), np.transpose(image_np[0],(1, 2, 0)))
            plt.imsave(os.path.join(gt_path, 'mask.png'), mask_np[0])
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


