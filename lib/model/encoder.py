import torch
from PIL import Image
from diffusers import AutoencoderKL
import torch.nn as nn
import numpy as np
from diffusers.models.unets.unet_2d_blocks import  DownBlock2D
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


class hidden_Encoder(nn.Module):
    '''
    encode by projection
    '''
    def __init__(self, patch_size=8, embed_dim=1280):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.db = nn.ModuleList([])
        db_list = [128, 256, 512, 768, 1280]
        for i, num in enumerate(db_list):
            self.db.append(DownBlock2D(
            num_layers=1,
            in_channels=num,
            out_channels=db_list[i+1] if i < len(db_list)-1 else 1280,
            temb_channels=512))

    def forward(self, model, image):  
            image = self.conv_in(image)
            t_emb = model.get_time_embed(sample=image, timestep=1000)
            emb = model.time_embedding(t_emb)
            for dblock in self.db:
                image, _ = dblock(image, emb)
            return image.flatten(-2).transpose(1, 2) 

if __name__ == '__main__':
    model = Encoder()
    image = torch.randn(16, 3, 64, 64).float()
    output = model(image)
    print(output.shape)