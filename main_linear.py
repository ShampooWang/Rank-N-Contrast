import argparse
import os
import sys
import logging
import torch
import time
from model import Encoder, model_dict
from dataset import *
from utils import *
from loss import PointwiseRankingLoss
from collections import defaultdict

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
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
    parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')
    parser.add_argument('--no_bias', action='store_true', help='Bias of linear regressor')

    # Others
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--ordinal_pretraining', type=bool, default=False)
    parser.add_argument('--orthog_basis', type=bool, default=False)
    parser.add_argument('--save_folder', type=str, default=None)

    opt = parser.parse_args()

    opt.model_name = 'Regressor_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_bias_{}_trial_{}'. \
        format(opt.dataset, opt.epochs, opt.learning_rate, opt.lr_decay_rate,
               opt.weight_decay, opt.momentum, opt.batch_size, not opt.no_bias, opt.trial)
    
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    
    if opt.save_folder is None:
        opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1]) # default is saving regressor to the encoder ckpt's folder
    else:
        os.makedirs(opt.save_folder, exist_ok=True)

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
    criterion = torch.nn.L1Loss()

    dim_in = model_dict[opt.model][1]
    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out, bias=not opt.no_bias)
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
    regressor = regressor.cuda()
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, regressor, criterion, pretrain_criterion


def train(train_loader, model, regressor, criterion, optimizer, epoch, opt, pretrain_criterion=None):
    model.eval()
    regressor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, returned_items in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = returned_items["img"].cuda(non_blocking=True)
        labels = returned_items["label"].cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = model(images)
            if opt.orthog_basis:
                feature_basis = returned_items["feature_basis"].cuda(non_blocking=True)
                features = features.norm(dim=-1, p=2).unsqueeze(1) * feature_basis
            elif pretrain_criterion is not None and hasattr(pretrain_criterion, "anchor"):
                scores = torch.matmul(features.double(), pretrain_criterion.anchor.unsqueeze(1)) # size [2B, 1]
                features = features * scores / features.norm(dim=-1, p=2).unsqueeze(1)

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


def validate(val_loader, model, regressor, opt, pretrain_criterion=None):
    model.eval()
    regressor.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for idx, returned_items in enumerate(val_loader):
            images = returned_items["img"].cuda(non_blocking=True)
            labels = returned_items["label"].cuda(non_blocking=True)
            bsz = labels.shape[0]
            features = model(images)

            if opt.orthog_basis:
                feature_basis = returned_items["feature_basis"].cuda(non_blocking=True)
                features = features.norm(dim=-1, p=2).unsqueeze(1) * feature_basis
            elif pretrain_criterion is not None and hasattr(pretrain_criterion, "anchor"):
                scores = torch.matmul(features.double(), pretrain_criterion.anchor.unsqueeze(1)) # size [2B, 1]
                features = features * scores / features.norm(dim=-1, p=2).unsqueeze(1)

            output = regressor(features)
            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses.avg


def main():
    opt = parse_option()
    set_seed(322)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, regressor, criterion, pretrain_criterion = set_model(opt)

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
        train(train_loader, model, regressor, criterion, optimizer, epoch, opt)

        valid_error = validate(val_loader, model, regressor, opt)
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
    test_loss = validate(test_loader, model, regressor, opt)
    to_print = 'Test L1 error: {:.3f}'.format(test_loss)
    print(to_print)


if __name__ == '__main__':
    main()
