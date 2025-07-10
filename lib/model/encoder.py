import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
import torch.nn as nn
import numpy as np
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        vae = AutoencoderKL.from_pretrained("../sd-vae-ft-mse")
        self.vae = vae.to("cuda") 

    def forward(self, image):
        with torch.no_grad():
            latent_dist = self.vae.encode(image).latent_dist  # 获取分布
            latent_sample = latent_dist.sample()  # 采样得到潜在特征
            # 或者使用均值：latent_sample = latent_dist.mode()
        return latent_sample 

if __name__ == '__main__':
    model = ImageEncoder()
    image = torch.randn(16, 1, 64, 64).float().to("cpu")
    image = image.repeat(1, 3, 1, 1)
    output = model(image)
    print(output.shape)