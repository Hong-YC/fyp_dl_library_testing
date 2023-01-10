import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

xy = np.load("../src/data/dataset/inputs.npz")
haha = xy['test_input']
print(xy.files)
# print(np.shape(haha))


# a dataset class template
class TrainDataset(Dataset):

    def __init__(self, transform = None):
        # path =  xxx
        xy = np.load(path) 

        self.n_samples
        self.x
        self.y
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    class ToTensor:
        def __call__(self, sample):
            inputs, targets = sample
            return torch.from_numpy(inputs), torch.from_numpy(targets)

    # dataset  = Traindataset(transform = ToTensor())