import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import set_seed
from datasets import AgeDB

to_tensor = transforms.Compose([transforms.ToTensor()])

class AgeDB_getpath(AgeDB):
    def __init__(self, aug_data_root, data_folder, seed, aug=None, split='train', noise_scale=0, two_view_aug=False, use_fix_aug=False, **kwargs):
        super().__init__(data_folder, seed, aug, split, noise_scale, two_view_aug, use_fix_aug, **kwargs)
        self.aug_data_root = aug_data_root

    def __getitem__(self, index):
        self.set_seed()
        row = self.df.iloc[index]
        path = os.path.join(self.data_folder, row['path'])
        img_name = row['path'].split("/")[-1].split(".")[0]
        img_dir = os.path.join(self.aug_data_root, img_name)
        os.makedirs(img_dir, exist_ok=True)
        aug_img = self.transform(Image.open(path).convert('RGB'))
        for i, img in enumerate(aug_img):
            if np.array(img).shape != (224, 224, 3):
                print(os.path.join(img_dir, f"view{i}.png"))
                print(np.array(img).shape)
                exit(0)
            img.save(os.path.join(img_dir, f"view{i}.png"))

        return len(aug_img)

    def get_transform(self, aug):
        
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

            # transforms_list.append(transforms.ToTensor())
            transform = transforms.Compose(transforms_list)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

        return self.get_two_view_aug(transform)
    
def main(aug_data_root):
    set_seed(322)
    dataset = AgeDB_getpath(
        aug_data_root=aug_data_root,
        data_folder="./datasets/AgeDB",
        seed=322,
        aug='crop,flip,color,grayscale',
        split="train",
        two_view_aug=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    
    total = 0
    for img_num in tqdm(dataloader):
        total += img_num.sum()

    print(f"Total augmented image number: {total}")

if __name__ == "__main__":
    main("/tmp2/jeffwang/Rank-N-Contrast/datasets/AgeDB/AgeDB_aug")