import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# xy = np.load("../src/data/dataset/inputs.npz")
# ypath = "../src/data/dataset/ground_truths.npz"
# y = np.load(ypath)
# haha = xy['test_input']
# # print(np.shape(haha)[0])
# print(y.files)


# a dataset class template
class TrainDataset(Dataset):

    def __init__(self, transform = None):
        # path =  xxx
        xpath = "../src/data/dataset/inputs.npz"
        ypath = "../src/data/dataset/ground_truths.npz"
        x = np.load(xpath)
        y = np.load(ypath)
        self.x = x['test_input']
        self.y = y['test_output']
        self.n_samples = np.shape(self.x)[0]
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

dataset  = TrainDataset(transform = ToTensor())
first_data = dataset[1]
features, labels = first_data
# print(features)
# print(labels)
print(len(dataset))