import argparse
import os
import sys
import logging
import torch
import time
from model import Encoder, model_dict
from utils import *
from exp_utils import plot_2d_umap
from loss import PointwiseRankingLoss
import numpy as np
import pandas as pd
from torch.utils import data
from scipy.stats import ortho_group


print = logging.info

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
    parser.add_argument('--no_bias', action='store_true', help='Bias of linear regressor')

    # Others
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--rank', type=int, default=512)

    opt = parser.parse_args()

    opt.model_name = 'Regressor_rank{}_bias_{}_trial_{}'. \
        format(opt.dataset, opt.rank, not opt.no_bias, opt.trial)
    
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]

    opt.save_folder = f'./checkpoints/concentric_circle/{opt.rank}_trial_{opt.trial}'
    os.makedirs(opt.save_folder, exist_ok=True)
    opt.umap_pic_path = os.path.join(opt.save_folder, "umap.png")

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, f'{opt.model_name}.log')),
            logging.StreamHandler()
        ]
    )

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt

class AgeDB(data.Dataset):
    def __init__(self, opt, data_folder, feature_rank=512, split='train'):
        df = pd.read_csv(f'./data/agedb.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder

        # Create basis with assigned rank 
        self.feature_rank = feature_rank
        self.feature_basis = torch.from_numpy(ortho_group.rvs(dim=feature_rank, random_state=opt.seed)).float()
        if self.feature_rank < 512:
            self.feature_basis = torch.cat((self.feature_basis, torch.zeros(self.feature_rank, 512-self.feature_rank)), dim=1)
        q, r = divmod(len(self.df) , self.feature_rank)
        self.feature_basis = torch.cat([self.feature_basis]*q + [self.feature_basis[:r]], dim=0)
        assert self.feature_basis.shape[0] == len(self.df)
        print(f"Initialize feature basis with rank: {self.feature_rank}, feature basis shape: {self.feature_basis.shape}")

        # Plot 2d umap for the created features
        if split == "test":
            labels = np.array(self.df['age'])
            plot_2d_umap(
                feats=labels[:, None] * self.feature_basis.numpy(),
                labels=labels,
                save_path=os.path.join(opt.save_folder, "test_uamp.png")
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        feature = self.feature_basis[index]

        return feature, label

def set_loader(opt):
    train_transform = get_transforms(split='val', aug=opt.aug) # close the augmentation during training
    val_transform = get_transforms(split='val', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")
    print(f"Val Transforms: {val_transform}")

    train_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, feature_rank=opt.rank, split='train')
    val_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, feature_rank=opt.rank, split='val')
    test_dataset = globals()[opt.dataset](opt, data_folder=opt.data_folder, feature_rank=opt.rank, split='test')

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
    criterion = torch.nn.L1Loss()
    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(512, dim_out, bias=not opt.no_bias)
    regressor = regressor.cuda()
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    return regressor, criterion


def train(train_loader, regressor, criterion, optimizer, epoch, opt):
    regressor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (features, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = features.cuda(non_blocking=True)

        output = regressor(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def validate(val_loader, regressor):
    regressor.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for idx, (features, labels) in enumerate(val_loader):
            features = features.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            output = regressor(features)
            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses.avg


def main():
    opt = parse_option()

    # Fixing random seed for the reproducibility
    set_seed(opt.seed)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    regressor, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, regressor)

    save_file_best = os.path.join(opt.save_folder, f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        regressor.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state['best_error']
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, regressor, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, regressor)
        print('Val L1 error: {:.3f}'.format(valid_error))

        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)
        print(f"Best Error: {best_error:.3f}")

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': regressor.state_dict(),
                'best_error': best_error
            }, save_file_best)

        torch.save({
            'epoch': epoch,
            'state_dict': regressor.state_dict(),
            'last_error': valid_error
        }, save_file_last)

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    regressor.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    test_loss = validate(test_loader, regressor)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)


if __name__ == '__main__':
    main()
