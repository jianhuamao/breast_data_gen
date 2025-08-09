import torch
from lib.model.Unet import load_unet_model
from diffusers import UNet2DConditionModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig
from lib.layer.lora import set_lora
def load_dict(model, config, optimizer=None):
    model_ckpt = torch.load(config.pretrain_model_path, weights_only=False)
    model.load_state_dict(model_ckpt['model_state_dict'])
    if config.lora_path is not None:
        lora_ckpt = torch.load(config.lora_path)
        model, unet_lora_layers = set_lora(lora_ckpt, model)
    if optimizer is not None:
        optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
        return {
            'model': model,
            'optimizer': optimizer
        }
    else:
        return {
            'model': model
        }
def load_model(config):
    if config.pretrain_model_path is not None:
        model = load_pretrain_model(config)
    else:
        if config.model_name == 'unet':
            model = load_unet_model(config)
    return model

def load_pretrain_model(config):
    if config.model_name == 'stabel_diffusion':
        unet = UNet2DConditionModel.from_pretrained(config.pretrain_model_path, subfolder="unet")
        text_encoder_cls = import_model_class_from_model_name_or_path(config.pretrain_model_path)
        text_encoder = text_encoder_cls.from_pretrained(
            config.pretrain_model_path, subfolder="text_encoder")
        tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model_path, subfolder="tokenizer")
        try:
            vae = AutoencoderKL.from_pretrained(
                config.pretrain_model_path, subfolder="vae")
        except OSError:
            vae = None
        # We only train the additional adapter LoRA layers
        if vae is not None:
            vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
    return {
        'unet': unet,
        'text_encoder': text_encoder,
        'vae': vae,
        'tokenizer': tokenizer,
    }

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def load_lora(config, unet, unet_lora_layers):
    lora_state_dict = torch.load(config.lora_path)
    unet_lora_layers.load_state_dict(lora_state_dict)
    for name, attn_processor in unet.attn_processors.items():
        if name in unet_lora_layers:
            attn_processor.load_state_dict(unet_lora_layers[name])
    return unet


if __name__ == '__main__':
    from lib.config import trainingConfig
    config = trainingConfig()
    config.pretrain_model_path = './my_model/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/stabel-diffusion'
    config.model_name = 'stabel_diffusion'
    model = load_model(config)
    print(model.keys())
