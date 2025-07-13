from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
# from ..model.encoder import ImageEncoder
from torch.utils.data.dataloader import default_collate

def collate_dict(batch):
    images   = default_collate([item['image']   for item in batch])
    gts      = default_collate([item['gt']      for item in batch])
    masks    = default_collate([item['mask']    for item in batch])
    return {
        'image': images,
        'gt': gts,
        'mask': masks,
    }
class MRIDataset(Dataset):
    def __init__(self, data_folder, training_file):
        self.data_folder = data_folder
        self.training_file = os.path.join(data_folder,training_file)
        with open(self.training_file, 'r') as f:
            lines = f.readlines()
            self.training_list = [line.strip() for line in lines]
            

    def __len__(self):
        return len(self.training_list)

    def get_sequnence_frame(self, image_sequence_path, gt_seqquence_path, mask_sequence_path):
        image = Image.open(image_sequence_path).convert('RGB')  # 'L' for grayscale
        gt = Image.open(gt_seqquence_path).convert('RGB')  # Use 'RGB' if color labels
        mask = Image.open(mask_sequence_path).convert('L').point(lambda p: 255 if p > 127 else 0, mode='L')
        # Convert to NumPy arrays
        image = np.array(image) / 255.0
        gt = np.array(gt) / 255.0
        mask = np.array(mask) / 255.0
        
        # crop to 64X64
        image = self.center_crop(image)
        gt = self.center_crop(gt)
        mask = self.center_crop(mask)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        gt = torch.tensor(gt, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return {'image': image,
                'gt': gt, 
                'mask': mask
                }

    def center_crop(self, img_array, target_size=64):
        h, w = img_array.shape[:2]       
        if h == w == target_size:
            return img_array

        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2

        if img_array.ndim == 2:             
            return img_array[start_h:start_h + target_size,
                            start_w:start_w + target_size]
        else:                                
            return img_array[start_h:start_h + target_size,
                            start_w:start_w + target_size, :]
    
    def __getitem__(self, idx):
        image_sequence_dir = os.path.join(self.data_folder,'b800', self.training_list[idx] + '_b800')
        gt_seqquence_dir = os.path.join(self.data_folder,'t1c', self.training_list[idx]+ '_t1c')
        mask_sequence_dir = os.path.join(self.data_folder,'segment', self.training_list[idx])
        frame_idx = random.choice(range(len(os.listdir(image_sequence_dir))))
        image_sequence_path = os.path.join(image_sequence_dir, '{}.png'.format(frame_idx))
        gt_seqquence_path = os.path.join(gt_seqquence_dir, '{}.png'.format(frame_idx))
        mask_sequence_path = os.path.join(mask_sequence_dir, '{}.png'.format(frame_idx))
        return self.get_sequnence_frame(image_sequence_path, gt_seqquence_path, mask_sequence_path=mask_sequence_path)


if __name__ == '__main__':
    data_folder = './data'
    dataset = MRIDataset(data_folder, "total_list.txt")
    indices = range(dataset.__len__())
    train_idx, eva_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eva_idx)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=4, sampler=train_sampler, collate_fn=collate_dict)
    eval_loader = DataLoader(dataset, batch_size=16, num_workers=4, sampler=eval_sampler, collate_fn=collate_dict)    
    print(len(train_loader))    
    # encoder = ImageEncoder()
    for data  in train_loader:
        print(data['mask'].shape)
        print(data['image'].min().item())
        print(data['mask'].min().item())
        print(data['gt'].min().item())
        plt.imsave('gt.png', data['gt'][0].permute(1, 2, 0).numpy())
        plt.imsave('image.png', data['image'][0].permute(1, 2, 0).numpy())
        # plt.imsave('mask.png', data['mask'][0].squeeze(1).numpy())
        
