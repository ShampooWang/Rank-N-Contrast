import argparse
import os
import sys
import logging
import torch
from torcheval.metrics import R2Score
import time
from model import Encoder, model_dict
from utils import *
# Data
import datasets
from torch.utils import data
# Others
import math
from tqdm import tqdm
from config.ordernamespace import OrderedNamespace
import yaml

print = logging.info

class BaseTask:
    def __init__(self):
        self.opt = None
        self.log_file_path = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def add_general_arguments(self, parser):
        parser.add_argument('--config', type=str, help='path to your .yaml configuration file')
        parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
        parser.add_argument('--resume', type=str, default=None, help='resume ckpt path')
        parser.add_argument('--ckpt', type=str, default='', help='path to the trained encoder')
        parser.add_argument('--seed', type=int, default=322)
        parser.add_argument('--save_folder', type=str, default=None)
        parser.add_argument("--fix_model_and_aug", action="store_true", help="Fixing the model parameters and augmentation random seed")
        parser.add_argument("--two_view_aug", action="store_true", help="Add two views of augmentation for training")

        return parser

    def parse_option(self, parser, log_file=True):
        parser = self.add_general_arguments(parser)    
        opt = parser.parse_args()
        config = yaml.load(open(opt.config, "r"), Loader=yaml.FullLoader)
        opt = OrderedNamespace([opt, config])

        if hasattr(self, "create_modelName_and_saveFolder"):
            opt = self.create_modelName_and_saveFolder(opt)
            if log_file:
                self.set_logging(log_file)
            print(f"Model name: {opt.model_name}")

        print(f"Options: {opt}")
        self.opt = opt

    def set_logging(self, log_file=True):
        handlers = [ logging.StreamHandler() ]
        if log_file and self.log_file_path is not None:
            handlers.append(logging.FileHandler(self.log_file_path)) 

        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=handlers
        )

    def set_seed(self):
        set_seed(self.opt.seed)
    
    def set_model(self):
        raise NotImplementedError
    
    def set_loader(self, batch_size, num_workers, two_view_aug=False):
        dataset = datasets.__dict__[self.opt.data.dataset]
        train_dataset = dataset(seed=self.opt.seed, split='train', **self.opt.data, two_view_aug=two_view_aug, use_fix_aug=self.opt.fix_model_and_aug)
        val_dataset = dataset(seed=self.opt.seed, split='val', **self.opt.data)
        test_dataset = dataset(seed=self.opt.seed, split='test', **self.opt.data)

        print(f'Train set size: {train_dataset.__len__()}')
        print(f'Valid set size: {val_dataset.__len__()}')
        print(f'Test set size: {test_dataset.__len__()}')

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    def set_optimizer(self):
        assert self.model is not None and self.criterion is not None
        parameters = list(self.model.parameters()) + list(self.criterion.parameters())
        config = self.opt.Encoder.optimizer
        optimizer_type = getattr(config, "type", "sgd")
        print(f"Using {optimizer_type} optimizer")

        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=config.learning_rate,
                                momentum=config.momentum, weight_decay=config.weight_decay)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        else:
            raise NotImplementedError(optimizer_type)

        self.optimizer = optimizer

    def adjust_learning_rate(self, config, epoch):
        lr = config.optimizer.learning_rate

        if hasattr(config.optimizer, "schedule"):
            for milestone in config.optimizer.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        elif hasattr(config.optimizer, "lr_decay_rate"):
            eta_min = lr * (config.optimizer.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / config.trainer.epochs)) / 2
        else:
            return
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
  
    def save_model(self, epoch, save_file, others: dict=None, verbose=True):
        if verbose:
            print('==> Saving...')
        state = {
            # 'opt': self.opt,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'criterion': self.criterion.state_dict()
        }
        if others is not None:
            state = {**state, **others}
        torch.save(state, save_file)
        del state

    def resume_status(self):
        if self.opt.resume is not None:
            ckpt_state = torch.load(self.opt.resume)
            self.model.load_state_dict(ckpt_state['model'])
            self.optimizer.load_state_dict(ckpt_state['optimizer'])
            self.start_epoch = ckpt_state['epoch'] + 1
            print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {self.opt.resume}!")
        else:
            self.start_epoch = 1

    def set_up_training(self, parser):
        # Only parse options for the first time
        if self.opt is None:
            self.parse_option(parser)
        self.set_seed()
        self.set_model_and_criterion()
        self.set_loader()
        self.set_optimizer()
        self.resume_status()

    def train_epoch(self, epoch):
        raise NotImplementedError
    
    def validation(self,):
        raise NotImplementedError
    
    def testing(self,):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError
