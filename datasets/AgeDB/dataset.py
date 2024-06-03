import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets.base_dataset import BaseDataset
import torch

print = logging.info


class AgeDB(BaseDataset):
    def __init__(self, data_folder, seed, aug=None, split='train', noise_scale=0.0, two_view_aug=False, use_fix_aug=False, **kwargs):
        super().__init__(split, data_folder, seed, two_view_aug)

        if split == "train" and noise_scale > 0:
            print(f"Adding gausain noise with scale {noise_scale} to the label.")
            df = pd.read_csv(f'./datasets/AgeDB/agedb_noise_scale_{noise_scale}.csv')
        else:
            df = pd.read_csv('./datasets/AgeDB/agedb_rank.csv')

        self.df = df[df['split'] == split]
        self.transform = self.get_transform(aug)
        self.use_fix_aug = use_fix_aug and split == "train"
        self.aug_dir = "./datasets/AgeDB/AgeDB_aug/"
        
        # if self.use_fix_aug:
        #     normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         normalize
        #     ])
        #     print(f"Fixing data augmentation. Loading augmented image from {self.aug_dir}")

    def __getitem__(self, index):
        # self.set_seed()
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)

        # if self.use_fix_aug:
        #     img_name = row['path'].split("/")[-1].split(".")[0]
        #     img_dir = os.path.join(self.aug_dir, img_name)
        #     imgs = [ Image.open(os.path.join(img_dir, f"view{i}.png")).convert('RGB') for i in range(2)]
        #     if self.two_view_aug:
        #         img = [ self.transform(_img) for _img in imgs ]
        #     else:
        #         img = self.transform(imgs[0])
        # else:
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        img = self.transform(img)

        return_dict = {
            "img": img,
            "label": label,
        }
        if "rank" in row:
            return_dict["rank"] = np.asarray([row['rank']]).astype(np.float32)

        return return_dict

    def get_transform(self, aug):
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if self.split == 'train':
            aug_list = aug.split(',')
            transforms_list = []

            if 'crop' in aug_list:
                transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
            else:
                transforms_list.append(transforms.Resize(256))
                transforms_list.append(transforms.CenterCrop(224))

            if 'flip' in aug_list:
                transforms_list.append(transforms.RandomHorizontalFlip())

            if 'color' in aug_list:
                transforms_list.append(transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8))

            if 'grayscale' in aug_list:
                transforms_list.append(transforms.RandomGrayscale(p=0.2))

            transforms_list.append(transforms.ToTensor())
            transforms_list.append(normalize)
            transform = transforms.Compose(transforms_list)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        return self.get_two_view_aug(transform)