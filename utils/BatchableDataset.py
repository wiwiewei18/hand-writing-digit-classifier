import torch
from torch.utils.data import Dataset


class BatchableDataset(Dataset):
    def __init__(self, filepath):
        self.image, self.target = torch.load(filepath)
        self.image = self.image / 255.0

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, ix):
        return self.image[ix], self.target[ix]
