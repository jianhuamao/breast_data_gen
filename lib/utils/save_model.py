from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from accelerate import Accelerator
def save_model_hook(models, unet_lora_layers, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        unet_keys = unet_lora_layers.state_dict().keys()

        for model in models:
            state_dict = model.state_dict()

            if state_dict.keys() == unet_keys:
                # unet
                unet_lora_layers_to_save = state_dict

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        LoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )
