import torch
import torch.nn as nn

class UNetWithSigmoid(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, sample, timesteps, **kwargs):
        out = self.unet(sample, timesteps, **kwargs)[0] 
        return self.sigmoid(out)
