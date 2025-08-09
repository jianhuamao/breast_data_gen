import argparse
import os
from lib.config import trainingConfig
from lib.data.dataset import MRIDataset, transform, FixedBatchSampler
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import diffusers
from lib.model.UnetWithSigmoid import UNetWithSigmoid
import matplotlib.pyplot as plt
from lib.train.train import train_loop
from lib.data.train_sampler import trainSampler
from log.log import train_log
from lib.utils.load_model import load_dict, load_model
from lib.layer.lora import set_lora
import swanlab
from lib.model.warm_up import lr_convert
import itertools
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--train_model', type=str, default='single')
    parser.add_argument('--nproc_per_node',type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_sampler', type=int , default= 1100)
    parser.add_argument('--num_train_timesteps', type=int, default= 1000)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--pretrain_model_path', type=str, default= None)
    parser.add_argument('--lora_path', type=str, default= None)
    parser.add_argument('--isDebug', type=bool, default= False)
    parser.add_argument('--sd', type=bool, default= False)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #GPU
    device = torch.device("cuda:0")
    #loader config
    config = trainingConfig(
        name = args.name,
        model_name = args.model_name,
        image_size=args.image_size,
        data_folder=args.data_folder,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        nproc_per_node=args.nproc_per_node,
        in_channels=args.in_channels,
        num_sampler = args.num_sampler,
        num_train_timesteps = args.num_train_timesteps,
        start_epoch = args.start_epoch,
        pretrain_model_path = args.pretrain_model_path,
        lora_path = args.lora_path,
        isDebug = args.isDebug,
        device = args.device,
        sd = args.sd,
        rank = args.rank,
        train_text_encoder = False,
        )
    if config.isDebug:
        project = 'test'
    else:
        project = 'breast_data_gen'
    swanlab.init(
    project=project,
    description=config.name,
    config={
        "learning_rate": config.lr,
        "architecture": "unet",
        "dataset": "t1c",
        "epochs": 500,
        'device': config.device
    }
    )
    #load data
    transforms = transform()
    train_dataset = MRIDataset(config.data_folder, "train_list.txt", transforms=transforms)
    eval_dataset = MRIDataset(config.data_folder, 'eval_list.txt', transforms=transforms)
    # train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=5000)
    if config.isDebug:
        sampler = FixedBatchSampler(train_dataset, batch_size=config.train_batch_size, num_batches=10)
        train_dataloader = DataLoader(train_dataset, num_workers=4, batch_sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=4, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, num_workers=4, shuffle=False, drop_last=True)
    #load_model
    model = load_model(config)
    unet = model['unet']
    #set lora
    if config.model_name == 'stabel_diffusion':
        unet, unet_lora_layers = set_lora(config, unet)
        model['unet'] = unet
        text_encoder_lora_layers = None
    #set optimizer
    if config.model_name == 'stabel_diffusion':
        params_to_optimize = (
            itertools.chain(unet_lora_layers.parameters(), text_encoder_lora_layers.parameters())
            if config.train_text_encoder
            else unet_lora_layers.parameters()
        )
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            lr=config.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
    )
        model['unet_lora_layers'] = unet_lora_layers
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if config.start_epoch > 0 and config.pretrain_model_path is not None:
        loaded_model = load_dict(model=unet, config = config, optimizer=optimizer)
        model = loaded_model['model']
        optimizer = loaded_model['optimizer']
        config.lr = optimizer.param_groups[0]["lr"]
    noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    lr_c = lr_convert(config=config, optimizer=optimizer)
    scheduler = lr_c.get_schedulers()
    train_loop(config, model, noise_scheduler, optimizer, scheduler, train_dataloader, eval_dataloader, swanlab, device=device)      
if __name__ == "__main__":
    main()
