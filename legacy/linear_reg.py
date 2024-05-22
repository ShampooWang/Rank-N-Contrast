import argparse
import os
import sys
import logging
import numpy as np
import torch
import time
from model import Encoder, model_dict
from dataset import *
from utils import *
from loss import PointwiseRankingLoss
from sklearn.linear_model import LinearRegression, Ridge
 
print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
    parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')

    # Others
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--ordinal_pretraining', type=bool, default=False)
    parser.add_argument('--orthog_basis', type=bool, default=False)

    opt = parser.parse_args()

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    print(f"Options: {opt}")

    return opt

class AgeDB(data.Dataset):
    def __init__(self, opt, data_folder, transform=None, split='train', feature_dim=512):
        df = pd.read_csv(f'./data/agedb.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform

        if getattr(opt, "orthog_basis", False):
            from scipy.stats import ortho_group
            self.feature_basis = torch.from_numpy(ortho_group.rvs(dim=feature_dim, random_state=opt.seed)).float()
            q, r = divmod(len(self.df) , feature_dim)
            self.feature_basis = torch.cat([self.feature_basis]*q + [self.feature_basis[:r]], dim=0)
            assert self.feature_basis.shape[0] == len(self.df)
            print(f"Initialize feature basis with rank: {feature_dim}, feature basis shape: {self.feature_basis.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        returned_items = {
            "img": img,
            "label": label,
        }

        if hasattr(self, "feature_basis"):
            returned_items["feature_basis"] = self.feature_basis[index]
        
        return returned_items
    

def set_loader(opt):
    train_transform = get_transforms(split='train', aug=opt.aug)
    val_transform = get_transforms(split='val', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")
    print(f"Val Transforms: {val_transform}")

    train_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, transform=train_transform, split='train')
    val_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, transform=val_transform, split='val')
    test_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, transform=val_transform, split='test')

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def set_model(opt):
    model = Encoder(name=opt.model)
    regressor = LinearRegression()
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    # Loading parameters for ordinal-type pretraining
    if 'criterion' in ckpt:
        if opt.ordinal_pretraining:
            pretrain_criterion  = PointwiseRankingLoss(objective="ordinal")
        else:
            pretrain_criterion = PointwiseRankingLoss()
        pretrain_criterion.load_state_dict(ckpt['criterion'])
        pretrain_criterion.cuda()
    else:
        pretrain_criterion = None

    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, regressor, pretrain_criterion


def train(train_loader, model, regressor, opt, pretrain_criterion=None):
    model.eval()
    X, y = [], []
    for returned_items in train_loader:
        images = returned_items["img"].cuda(non_blocking=True)
        labels = returned_items["label"].cuda(non_blocking=True).squeeze(1)

        with torch.no_grad():
            features = model(images)
            if opt.orthog_basis:
                feature_basis = returned_items["feature_basis"].cuda(non_blocking=True)
                features = features.norm(dim=-1, p=2).unsqueeze(1) * feature_basis
            elif pretrain_criterion is not None and hasattr(pretrain_criterion, "anchor"):
                scores = torch.matmul(features.double(), pretrain_criterion.anchor.unsqueeze(1)) # size [2B, 1]
                features = features * scores / features.norm(dim=-1, p=2).unsqueeze(1)

        X.append(features.detach().cpu().numpy())
        y.append(labels.detach().cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    reg = regressor.fit(X, y)
    train_mae = np.abs(reg.predict(X) - y).mean(0)

    print(f"Training MAE: {train_mae}")

    return reg


def validate(val_loader, model, regressor, opt, pretrain_criterion=None):
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for returned_items in val_loader:
            images = returned_items["img"].cuda(non_blocking=True)
            labels = returned_items["label"].cuda(non_blocking=True).squeeze(1)
            features = model(images)

            if opt.orthog_basis:
                feature_basis = returned_items["feature_basis"].cuda(non_blocking=True)
                features = features.norm(dim=-1, p=2).unsqueeze(1) * feature_basis
            elif pretrain_criterion is not None and hasattr(pretrain_criterion, "anchor"):
                scores = torch.matmul(features.double(), pretrain_criterion.anchor.unsqueeze(1)) # size [2B, 1]
                features = features * scores / features.norm(dim=-1, p=2).unsqueeze(1)

            X.append(features.detach().cpu().numpy())
            y.append(labels.detach().cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    valid_mae = np.abs(regressor.predict(X) - y).mean(0)

    return valid_mae


def main():
    opt = parse_option()
    set_seed(322)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, regressor, pretrain_criterion = set_model(opt)
    regressor = train(train_loader, model, regressor, opt, pretrain_criterion)
    print(f"Validation MAE: {validate(val_loader, model, regressor, opt, pretrain_criterion)}") 
    print(f"Testing MAE: {validate(test_loader, model, regressor, opt, pretrain_criterion)}") 

if __name__ == '__main__':
    main()
