import torch
import numpy as np 
import albumentations
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class subDataset(Dataset):
    def __init__(self, dataset, alpha):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return int(self.alpha*len(self.dataset))

    def __getitem__(self, i):
        return self.dataset[i]


class NoisySet(Dataset):
    def __init__(self, dataset, sigma = 0):
        self.sigma = sigma
        self.data = []
        for i in range(len(dataset)):
            original_img = dataset[i][0] 
            label = dataset[i][1] 
            new_img = self.apply_noise(original_img, sigma=self.sigma)

            self.data.append((new_img, label))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def apply_noise(self, img_arr, sigma = 0.1):
        mean = 0
        noise = torch.tensor(np.random.normal(mean, sigma, img_arr.shape), dtype=torch.float)
        return img_arr + noise

class NaturalImageDataset(Dataset):
    def __init__(self, path, labels, resize=84, tfms=None):
        self.img_dim = resize
        img_dim = self.img_dim
        self.X = path
        self.y = labels
        # apply augmentations
        if tfms == 0: # if validating
            self.aug = albumentations.Compose([
                albumentations.Resize(img_dim, img_dim, always_apply=True),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], always_apply=True)
            ])
        else: # if training
            self.aug = albumentations.Compose([
                albumentations.Resize(img_dim, img_dim, always_apply=True),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], always_apply=True)
            ])    
            self.aug = albumentations.Compose([
                albumentations.Resize(img_dim, img_dim, always_apply=True),
                albumentations.HorizontalFlip(p=0.8),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.20,
                    scale_limit=0.20,
                    rotate_limit=0,
                    p=0.8
                ),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], always_apply=True)
            ])

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

