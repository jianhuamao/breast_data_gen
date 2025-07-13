from dataclasses import dataclass

@dataclass
class trainingConfig:
    model: str = "DDPM"
    dataset: object = None
    image_size: int = 64
    train_batch_size: int = 16
    eval_batch_size: int = 16
    lr: float = 1e-5
    num_epochs: int = 100
    data_folder: str = "data"
    nproc_per_node: int = 4
    in_channels: int = 3
    num_sampler: int = 1100
    num_train_timesteps: int = 1000
    start_epoch: int = 0
    pretrain_model_path: str = None