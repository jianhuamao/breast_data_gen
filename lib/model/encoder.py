import torch
from PIL import Image
from diffusers import AutoencoderKL
import torch.nn as nn
import numpy as np
# encoder by vae
# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super(ImageEncoder, self).__init__()
#         vae = AutoencoderKL.from_pretrained("./sd-vae-ft-mse")
#         self.vae = vae

#     def forward(self, image):
#         with torch.no_grad():
#             latent_dist = self.vae.encode(image).latent_dist  # 获取分布
#             latent_sample = latent_dist.sample()  # 采样得到潜在特征
#             # 或者使用均值：latent_sample = latent_dist.mode()
#         # latent_sample = latent_sample.flatten(start_dim=-2)
#         return latent_sample 


class ImageEncoder(nn.Module):
    '''
    encode by projection
    '''
    def __init__(self, patch_size=8, embed_dim=1280):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(3*patch_size*patch_size, embed_dim)

    def forward(self, x):                # x: [B,3,H,W]
        B, C, H, W = x.shape
        x = x.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        x = x.permute(0,2,3,1,4,5).contiguous()  # [B, h, w, C, ps, ps]
        x = x.view(B, -1, C*self.ps*self.ps)     # [B, N, C*ps²]
        return self.proj(x) 
if __name__ == '__main__':
    model = ImageEncoder()
    image = torch.randn(16, 3, 64, 64).float()
    output = model(image)
    print(output.shape)