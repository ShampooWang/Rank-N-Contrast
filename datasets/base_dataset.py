
from torch.utils import data
from utils import TwoCropTransform
import torch
import random
import logging
import numpy as np

print = logging.info

class BaseDataset(data.Dataset):
    def __init__(self, split: str, data_folder: str, seed: int, two_view_aug: bool=False):
        self.df = None
        self.split = split
        self.data_folder = data_folder
        self.seed = seed
        self.two_view_aug = two_view_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        raise NotImplementedError
    
    def get_data_occurences(self):
        labels, occurence = np.unique(self.df["age"], return_counts=True)
        return { age: occ for (age, occ) in zip(labels, occurence) }

    def set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def get_two_view_aug(self, transform):
        """Get two views of augmentation if two_view_aug and split is train, else remain the original single view

        Returns:
            torchvision.Transform
        """
        if self.split == "train" and self.two_view_aug:
            print("Using two views of data augmentation for training")
            return TwoCropTransform(transform)
        else:
            return transform

    def get_transform(self):
        raise NotImplementedError