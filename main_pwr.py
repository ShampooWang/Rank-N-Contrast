import argparse
import os
import sys
import logging
import torch
import time
import wandb

from dataset import *
from utils import *
from model import Encoder
from loss import PointwiseRankingLoss

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--projection', type=bool, default=False, help="Whether to add linear prjectors on the top of the model")
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')

    parser.add_argument('--wandb', type=bool, default=True, help='Using wandb to recored the experiments')

    # RnCLoss Parameters
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--feature_norm', type=str, default='l1', choices=['l1', 'l2'], help='Norm of the features')
    parser.add_argument('--objective', type=str, default='l2', choices=['l1', 'l2', 'covariance', 'correlation', 'ordinal'], help='Objective funtion of pointwise ranking')

    opt = parser.parse_args()

    opt.model_path = './checkpoints/{}_models'.format(opt.dataset)
    opt.model_name = 'PwR_{}_{}_ep_{}_norm_{}_obj_{}_proj_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.epochs, opt.feature_norm, opt.objective, opt.projection, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):
    train_transform = get_transforms(split='train', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")

    train_dataset = globals()[opt.dataset](
        data_folder=opt.data_folder,
        transform=TwoCropTransform(train_transform),
        split='train'
    )
    print(f'Train set size: {train_dataset.__len__()}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    return train_loader


def set_model(opt):
    model = Encoder(name=opt.model, projection=opt.projection)
    criterion = PointwiseRankingLoss(feature_norm=opt.feature_norm, objective=opt.objective)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return model, criterion

def set_optimizer(opt, model, criterion):
    parameters = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.SGD(parameters, lr=opt.learning_rate,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)

    return optimizer

def save_model(model, optimizer, opt, epoch, save_file, criterion):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'criterion': criterion.state_dict()
    }
    torch.save(state, save_file)
    del state


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        images, labels = data_tuple
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()

    if hasattr(criterion, "weight"):
        print(f"Learnable weight: {criterion.weight.exp().item()}")

    if hasattr(criterion, "bias"):
        print(f"Learnable bias: {criterion.bias.exp().item()}")

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, criterion)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, opt)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file, criterion)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder, 'curr_last.pth')
            save_model(model, optimizer, opt, epoch, save_file, criterion)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file, criterion)


if __name__ == '__main__':
    main()
