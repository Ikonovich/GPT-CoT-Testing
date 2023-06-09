import pandas as pd
from torch.utils.data import Dataset


class JsonDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]["Question"]
        y = self.data[item]["GT"]
        original = self.data[item]
        return x, y, original

