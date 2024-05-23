import os
from torchvision import transforms
import math
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import random

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
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

    return transform


def get_label_dim(dataset):
    if dataset in ['AgeDB', 'IMDBWIKI']:
        label_dim = 1
    else:
        raise ValueError(dataset)
    return label_dim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.optimizer.learning_rate
    eta_min = lr * (args.optimizer.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.trainer.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.optimizer.learning_rate,
                                momentum=opt.optimizer.momentum, weight_decay=opt.optimizer.weight_decay)

    return optimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_encoder_and_regressor_params():
    from model import Encoder, SupResNet, model_dict
    set_seed(322)
    model_dir = "/tmp2/jeffwang/Rank-N-Contrast/checkpoints/seed322"
    os.makedirs(model_dir, exist_ok=True)
    model_list = ["resnet18", "resnet50"]

    def save_model(model: torch.nn.Module, save_file: str):
        torch.save(
            {"model":model.state_dict()}, save_file
        )
        del model

    # for model_name in model_list:
    #     save_model(Encoder(model_name), os.path.join(model_dir, f"{model_name}.pth"))

    for model_name in model_list:
        save_model(SupResNet(model_name, num_classes=1), os.path.join(model_dir, f"sup{model_name}.pth"))

    # for model_name in model_list:
    #     dim_in = model_dict[model_name][1]
    #     dim_out = 1
    #     model = torch.nn.Linear(dim_in, dim_out)
    #     save_model(model, os.path.join(model_dir, f"{model_name}_regressor.pth"))

if __name__ == "__main__":
    create_encoder_and_regressor_params()