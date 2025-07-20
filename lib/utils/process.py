# image_mean = 0.0990 image_std = 0.1030  gt_mean = 0.1016 gt_std = 0.1430
from torchvision import transforms

class transform():
    def __init__(self):
        self.train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop((256, 256), padding=4, padding_mode='reflect'),
        transforms.Normalize([0.0990], [0.1030])
        ])
        self.train_gt_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop((256, 256), padding=4, padding_mode='reflect'),
            transforms.Normalize([0.1016], [0.1430])
        ])
        self.train_mask_transform =transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop((256, 256), padding=4, padding_mode='reflect'),
        ])
    def get_image_transform(self):
        return self.train_image_transform
    def get_gt_transform(self):
        return self.train_gt_transform
    def get_mask_transform(self):
        return self.train_mask_transform