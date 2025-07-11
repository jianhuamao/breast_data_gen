from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from ..model.encoder import ImageEncoder
class MRIDataset(Dataset):
    def __init__(self, data_folder, training_file):
        self.data_folder = data_folder
        self.training_file = os.path.join(data_folder,training_file)
        with open(self.training_file, 'r') as f:
            lines = f.readlines()
            self.training_list = [line.strip() for line in lines]
            

    def __len__(self):
        return len(self.training_list)

    def get_sequnence_frame(self, image_sequence_path, label_seqquence_path):
        image = Image.open(image_sequence_path).convert('RGB')  # 'L' for grayscale
        label = Image.open(label_seqquence_path).convert('RGB')  # Use 'RGB' if color labels

        # Convert to NumPy arrays
        image = np.array(image) / 255.0
        label = np.array(label) / 255.0
        # crop to 64X64
        image = self.center_crop(image)
        label = self.center_crop(label)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        return image, label

    def center_crop(self, img_array, target_size=64):
        h, w, C = img_array.shape
        if h == w == target_size:
            return img_array  
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img_array[start_h:start_h + target_size, start_w:start_w + target_size, :]
    
    def __getitem__(self, idx):
        image_sequence_dir = os.path.join(self.data_folder,'b800', self.training_list[idx] + '_b800')
        label_seqquence_dir = os.path.join(self.data_folder,'t1c', self.training_list[idx]+ '_t1c')
        frame_idx = random.choice(range(len(os.listdir(image_sequence_dir))))
        image_sequence_path = os.path.join(image_sequence_dir, '{}.png'.format(frame_idx))
        label_seqquence_path = os.path.join(label_seqquence_dir, '{}.png'.format(frame_idx))
        return self.get_sequnence_frame(image_sequence_path, label_seqquence_path)


if __name__ == '__main__':
    data_folder = './data'
    dataset = MRIDataset(data_folder, "total_list.txt")
    indices = range(dataset.__len__())
    train_idx, eva_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eva_idx)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=0, sampler=train_sampler)
    eval_loader = DataLoader(dataset, batch_size=16, num_workers=0, sampler=eval_sampler)    
    print(len(train_loader))    
    # encoder = ImageEncoder()
    for image, label in eval_loader:
        print(image.shape)
        # plt.imsave('test.png', label[0].permute(1, 2, 0).numpy())