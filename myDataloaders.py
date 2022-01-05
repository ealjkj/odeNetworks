import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 


class subDataset(Dataset):
    def __init__(self, dataset, alpha):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return int(self.alpha*len(pretrainset))

    def __getitem__(self, i):
        return dataset[i]


class NoisySet(Dataset):
    def __init__(self, dataset, sigma = 0):
        self.sigma = sigma
        self.data = []
        for i in range(len(dataset)):
            self.data.append((self.apply_noise(dataset[i][0], sigma=self.sigma), dataset[i][1]))
        
    def __len__(self):
        return len(pretestset)//100

    def __getitem__(self, i):
        return self.data[i]

    def apply_noise(img_arr, sigma = 0.1):
        mean = 0
        noise = torch.tensor(np.random.normal(mean, sigma, img_arr.shape), dtype=torch.float)
        return img_arr + noise


