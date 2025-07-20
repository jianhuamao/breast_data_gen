import torch

def load_dict(model, pretrain_model_path, optimizer=None):
    ckpt = torch.load(pretrain_model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return {
            'model': model,
            'optimizer': optimizer
        }
    else:
        return {
            'model': model
        }