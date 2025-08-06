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
# from lib.utils.process import train_transform
from torchvision import transforms
import albumentations as A
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
    def __init__(self, data_folder, sequence_list_txt, transforms=None):
        self.data_folder = data_folder
        self.transforms = transforms
        self._should_reinit_norm = True
        if transforms is not None:
            self.image_norm = transforms.image_norm
            self.gt_norm = transforms.gt_norm
            self.mask_to_tensor = transforms.mask_to_tensor
            if 'train' in sequence_list_txt:
                try:
                    self.train_tf = transforms.train_transform
                except:
                    self.train_tf = None
            else: self.train_tf = None
        self.image_path_list = []
        self.sequence_path = os.path.join(data_folder,sequence_list_txt)
        with open(self.sequence_path, 'r') as f:
            lines = f.readlines()
            self.sequence_list = [line.strip() for line in lines]
        for idx in self.sequence_list:
            image_sequence_dir = os.path.join(self.data_folder,'b800', idx + '_b800')
            for i in os.listdir(image_sequence_dir):
                self.image_path_list.append(os.path.join(image_sequence_dir, i))        

    def __len__(self):
        return len(self.image_path_list)

    def get_sequnence_frame(self, image_sequence_path, gt_seqquence_path, mask_sequence_path):
        image = Image.open(image_sequence_path).convert('RGB')  
        gt = Image.open(gt_seqquence_path).convert('RGB')  
        mask = Image.open(mask_sequence_path).convert('L').point(lambda p: 255 if p > 127 else 0, mode='L')
        # Convert to NumPy arrays
        image = np.array(image) / 255.0
        gt = np.array(gt) / 255.0
        mask = np.array(mask) / 255.0
        if self._should_reinit_norm:
            self.reinit_norm(image)
        if self.transforms is not None:
            if self.train_tf is not None:
                aug = self.train_tf(image=image, gt=gt, mask=mask)
                image = aug['image']
                gt = aug['gt']
                mask = aug['mask']
            image = self.image_norm(image=image)
            gt = self.gt_norm(image=gt)
            mask = self.mask_to_tensor(image=mask)
            return {
                'image': image['image'].float(),
                'gt': gt['image'].float(),
                'mask': mask['image'].float()
            }
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            gt = torch.tensor(gt, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return {
                'image': image,
                'gt': gt,
                'mask': mask
            }
    def __get_seed__(self):
        return np.random.randint(2147483647)
    def center_crop(self, img_array, target_size=256):
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
        image_path = self.image_path_list[idx]
        t1c_path = image_path.replace('b800', 't1c')
        mask_path = image_path.replace('b800', 'segment')
        mask_path = mask_path.replace('_segment', '')
        return self.get_sequnence_frame(image_path, t1c_path, mask_sequence_path=mask_path)
    def reinit_norm(self, img):
        self.transforms.reinit_norm(img)
        self.image_norm = self.transforms.image_norm
        self.gt_norm = self.transforms.gt_norm
        self._should_reinit_norm = False


class transform():
    '''
    default norm_size = (64, 64, 1)
    '''
    def __init__(self):
        self.img_dim = 3
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.25),
            A.Rotate(limit=15, p=0.25)
        ], additional_targets={'gt': 'image', 'mask': 'mask'})
        self.image_norm = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0),
            A.ToTensorV2()
            ])
        self.gt_norm = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0),
            A.ToTensorV2()
            ])
        self.image_norm = A.ToTensorV2()
        self.gt_norm = A.ToTensorV2()
        self.mask_to_tensor = A.ToTensorV2()
    def get_dimension(self, img):
        dim = img.shape[-1]
        return dim
        
    def reinit_norm(self, img):
        dim = self.get_dimension(img)
        if dim != 1:
            self.image_norm = A.Compose([
                A.Normalize(mean=[0.5]*dim, std=[0.5]*dim, max_pixel_value=1.0),
                A.ToTensorV2()
                ])
            self.gt_norm = A.Compose([
                A.Normalize(mean=[0.5]*dim, std=[0.5]*dim, max_pixel_value=1.0),
                A.ToTensorV2()
                ])
            print('12')
        else: pass

if __name__ == '__main__':
    transform = transform()
    data_folder = './data'
    train_dataset = MRIDataset(data_folder, "train_list.txt", transforms=transform)
    eval_dataset = MRIDataset(data_folder=data_folder, sequence_list_txt='eval_list.txt', transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8, num_workers=4,shuffle=False)
    train_len = len(train_dataset)  
    print(train_len)  
    # encoder = ImageEncoder()
    for data in eval_loader:
        print('image:',data['image'].shape)
        print('gt',data['gt'].shape)
        print('mask',data['mask'].shape)
        breakpoint()
    
