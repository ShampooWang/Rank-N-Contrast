import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data


class AgeDB(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train', noise_scale=0.0):
        if split == "train" and noise_scale > 0:
            print(f"Adding gausain noise with scale {noise_scale} to the label.")
            df = pd.read_csv(f'./data/agedb_noise_scale_{noise_scale}.csv')
        else:
            df = pd.read_csv(f'./data/agedb.csv')

        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label
