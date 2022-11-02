from torch.utils.data.dataset import Dataset
from skimage.io import imread
import numpy as np
import pandas as pd
import os

import torch
import torchvision as tv


class CelebaEmbedDataset(Dataset):
    def __init__(self, npy_path, csv_path):
        super().__init__()
        self.npy_path = npy_path
        self.csv_path = csv_path

        self.embed = np.load(npy_path)
        df = pd.read_csv(csv_path).drop(columns=['image_id'])
        self.metrics = {col: torch.as_tensor(df[col]).float() for col in df.columns}

        self.num_items = len(npy_path)


    def __len__(self):
        return self.num_items


    def __getitem__(self, index):
        item = {col: values[index] for col, values in self.metrics.items()}
        item['embed'] = self.embed[index]

        return item
