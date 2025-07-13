'''
todo:
write a own diffusion Model
'''
 
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D

class UNet2DConditionModel(nn.Module):
    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Time embedding
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = nn.Linear(block_out_channels[0], time_embed_dim)
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        ch = block_out_channels[0]
        for i, (out_ch, block_type) in enumerate(zip(block_out_channels, down_block_types)):
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(
                self._make_down_block(block_type, ch, out_ch, layers_per_block, cross_attention_dim, time_embed_dim)
            )
            ch = out_ch

        # Mid block
        self.mid_block = self._make_mid_block(ch, cross_attention_dim, time_embed_dim)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        reversed_out_channels = list(reversed(block_out_channels))
        for i, (out_ch, block_type) in enumerate(zip(reversed_out_channels, up_block_types)):
            is_final = i == len(reversed_out_channels) - 1
            prev_ch = ch + reversed_out_channels[min(i+1, len(reversed_out_channels)-1)]
            self.up_blocks.append(
                self._make_up_block(block_type, prev_ch, out_ch, layers_per_block, cross_attention_dim, time_embed_dim)
            )
            ch = out_ch

        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def _make_down_block(self, block_type, in_ch, out_ch, layers, cross_attention_dim, time_embed_dim):
        # Simplified: ResNet + CrossAttn
        return nn.ModuleList([
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_channels=time_embed_dim),
            BasicTransformerBlock(
                out_ch, num_attention_heads=8, attention_head_dim=out_ch // 8,
                cross_attention_dim=cross_attention_dim
            ) if "CrossAttn" in block_type else None,
            Downsample2D(out_ch) if i == layers-1 else None
        ])

    def _make_mid_block(self, ch, cross_attention_dim, time_embed_dim):
        return nn.ModuleList([
            ResnetBlock2D(ch, ch, temb_channels=time_embed_dim),
            BasicTransformerBlock(ch, 8, ch // 8, cross_attention_dim),
            ResnetBlock2D(ch, ch, temb_channels=time_embed_dim),
        ])

    def _make_up_block(self, block_type, in_ch, out_ch, layers, cross_attention_dim, time_embed_dim):
        return nn.ModuleList([
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_channels=time_embed_dim),
            BasicTransformerBlock(
                out_ch, 8, out_ch // 8, cross_attention_dim
            ) if "CrossAttn" in block_type else None,
            Upsample2D(out_ch) if i == layers-1 else None
        ])

    def forward(self, sample, timestep, encoder_hidden_states):
        # 1. Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)

        # 2. Input projection
        x = self.conv_in(sample)

        # 3. Down blocks
        down_block_res_samples = (x,)
        for down_block in self.down_blocks:
            for layer in down_block:
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, t_emb)
                elif isinstance(layer, BasicTransformerBlock):
                    x = layer(x, encoder_hidden_states=encoder_hidden_states)
                elif isinstance(layer, Downsample2D):
                    x = layer(x)
            down_block_res_samples += (x,)

        # 4. Mid block
        for layer in self.mid_block:
            if isinstance(layer, ResnetBlock2D):
                x = layer(x, t_emb)
            elif isinstance(layer, BasicTransformerBlock):
                x = layer(x, encoder_hidden_states=encoder_hidden_states)

        # 5. Up blocks
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block):]
            down_block_res_samples = down_block_res_samples[:-len(up_block)]
            x = torch.cat([x, res_samples[0]], dim=1)  # Skip connection
            for layer in up_block:
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, t_emb)
                elif isinstance(layer, BasicTransformerBlock):
                    x = layer(x, encoder_hidden_states=encoder_hidden_states)
                elif isinstance(layer, Upsample2D):
                    x = layer(x)

        # 6. Output projection
        x = self.conv_out(x)
        return UNet2DConditionOutput(sample=x)