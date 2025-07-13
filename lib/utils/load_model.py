import torch

def load_dict(model, pretrain_model_path):
    ckpt = torch.load(pretrain_model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model