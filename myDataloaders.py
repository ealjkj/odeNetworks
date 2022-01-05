import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 


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
        return len(pretestset)//100

    def __getitem__(self, i):
        return self.data[i]

    def apply_noise(self, img_arr, sigma = 0.1):
        mean = 0
        noise = torch.tensor(np.random.normal(mean, sigma, img_arr.shape), dtype=torch.float)
        return img_arr + noise


