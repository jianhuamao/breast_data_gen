from dataclasses import dataclass

@dataclass
class trainingConfig:
    name: str = "train"
    model_name: str = 'unet'
    dataset: object = None
    image_size: int = 64
    train_batch_size: int = 16
    eval_batch_size: int = 16
    lr: float = 1e-4
    num_epochs: int = 100
    data_folder: str = "data"
    nproc_per_node: int = 4
    in_channels: int = 3
    num_sampler: int = 1100
    num_train_timesteps: int = 1000
    start_epoch: int = 0
    pretrain_model_path: str = None
    isDebug: bool = False
    device: str = "0"
    sd: bool =False
    rank: int = 16
    lora_path: str = None
    train_text_encoder: bool = False