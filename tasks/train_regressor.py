import argparse
import os
import sys
import logging
import torch
from torcheval.metrics import R2Score
import time
from model import Encoder, model_dict
from utils import *
from .base_task import BaseTask
# Data
import datasets
from torch.utils import data
# Others
import math
from tqdm import tqdm


print = logging.info


class FeatureLabel_pairs(data.Dataset):
    def __init__(self, features, labels) -> None:
        if isinstance(features, list):
            self.features = torch.cat(features, dim=0)
        else:
            self.features = features

        if isinstance(features, list):
            self.labels = torch.cat(labels, dim=0)
        else:
            self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class TrainRegressor(BaseTask):
    def __int__(self):
        super().__int__()
        self.encoder = None

    def create_modelName_and_saveFolder(self, opt):
        div_scale = getattr(opt.feature_extract, "div_scale", 1.0)
        normalized_by_D = getattr(opt.feature_extract, "normalized_by_D", False)
        opt.model_name = 'Regressor_{}_divScale_{}_normalied_D_{}_trial_{}'\
        .format(opt.data.dataset, div_scale, normalized_by_D, opt.trial)    
        
        if opt.resume is not None:
            opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
        
        if opt.save_folder is None:
            opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1]) # default is saving regressor to the encoder ckpt's folder
        else:
            os.makedirs(opt.save_folder, exist_ok=True)
    
        self.log_file_path = os.path.join(opt.save_folder, f'{opt.model_name}.log')

        return opt

    def set_model_and_criterion(self):
        encoder = Encoder(name=self.opt.Encoder.type)
        dim_in = model_dict[self.opt.Encoder.type][1]
        dim_out = get_label_dim(self.opt.data.dataset)
        regressor = torch.nn.Linear(dim_in, dim_out)
        ckpt = torch.load(self.opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']

        if torch.cuda.device_count() > 1:
            encoder.encoder = torch.nn.DataParallel(encoder.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

        if self.opt.fix_model_and_aug:
            model_path = f"./checkpoints/seed322/{self.opt.Encoder.type}_regressor.pth"
            model_params = torch.load(model_path)["model"]
            regressor.load_state_dict(model_params)
            print(f"Fixing regressor's initialization. Model checkpoint Loaded from {model_path}!")

        self.model = regressor.cuda()
        self.criterion  = torch.nn.L1Loss().cuda()
        self.encoder = encoder.cuda()
        self.encoder.load_state_dict(state_dict)
        print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {self.opt.ckpt}!")

    
    def get_encoder_features(self, dataset, shuffle=False):
        self.encoder.eval()
        config = self.opt.Regressor.trainer
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=False
        )
        div_scale = getattr(self.opt.feature_extract, "div_scale", 1.0)
        print(f"Start extracting frozen features. Divide feature by the scale: {div_scale}")
        normalized_by_D = getattr(self.opt.feature_extract, "normalized_by_D", False)
        if normalized_by_D: 
            print(f"Normalize features by sqrt(D)")
        
        Z = []
        y = []
        with torch.no_grad():
            for returned_items in tqdm(loader):
                images = returned_items["img"].cuda(non_blocking=True)
                features = self.encoder(images) / div_scale
                if normalized_by_D:
                    features = features / math.sqrt(features.shape[1])
                Z.append(features.cpu())
                y.append(returned_items["label"])
        del loader
        return FeatureLabel_pairs(Z, y)

    def set_loader(self):
        dataset = datasets.__dict__[self.opt.data.dataset]
        train_dataset = dataset(seed=self.opt.seed, split='train', **self.opt.data, two_view_aug=False, use_fix_aug=self.opt.fix_model_and_aug)
        val_dataset = dataset(seed=self.opt.seed, split='val', **self.opt.data)
        test_dataset = dataset(seed=self.opt.seed, split='test', **self.opt.data)

        if self.opt.Regressor.trainer.verbose:
            print(f'Train set size: {train_dataset.__len__()}')
            print(f'Valid set size: {val_dataset.__len__()}')
            print(f'Test set size: {test_dataset.__len__()}')

        batch_size = self.opt.Regressor.trainer.batch_size
        num_workers = self.opt.Regressor.trainer.num_workers
        self.train_loader = torch.utils.data.DataLoader(
            self.get_encoder_features(train_dataset, shuffle=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.get_encoder_features(val_dataset), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.get_encoder_features(test_dataset), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    def set_optimizer(self):
        assert self.model is not None and self.criterion is not None
        parameters = list(self.model.parameters())
        config = self.opt.Regressor.optimizer
        optimizer = torch.optim.SGD(
            parameters, 
            lr=config.learning_rate,
            momentum=config.momentum, 
            weight_decay=config.weight_decay
        )

        self.optimizer = optimizer

    def resume_status(self):
        if self.opt.resume is not None:
            ckpt_state = torch.load(self.opt.resume)
            self.model.load_state_dict(ckpt_state['model'])
            self.optimizer.load_state_dict(ckpt_state['optimizer'])
            self.start_epoch = ckpt_state['epoch'] + 1
            self.best_error = ckpt_state['best_error']
            print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {self.opt.resume}!")
        else:
            self.start_epoch = 1

    def train_epoch(self, epoch):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for idx, (features, labels) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            with torch.no_grad():
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            output = self.model(features.detach())
            loss = self.criterion(output, labels)
            losses.update(loss.item(), bsz)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if (idx + 1) % opt.Regressor.trainer.print_freq == 0:
            #     print('Train: [{0}][{1}/{2}]\t'
            #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
            #         epoch, idx + 1, len(loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses))
            #     sys.stdout.flush()

        if self.opt.Regressor.trainer.verbose:
            print('Epoch [{0}], average loss: {1.avg:.3f}'.format(epoch, losses))

        sys.stdout.flush()
        
    def validation(self, return_r2_score=False):
        self.model.eval()
        errors = AverageMeter()
        criterion_l1 = torch.nn.L1Loss()
        r2metric = R2Score()
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]
                output = self.model(features)
                error_l1 = criterion_l1(output, labels)
                errors.update(error_l1.item(), bsz)
                r2metric.update(output, labels)

        if return_r2_score:
            return errors.avg, r2metric.compute()
        else:
            return errors.avg

    
    def testing(self, return_r2_score=False):
        self.model.eval()
        errors = AverageMeter()
        criterion_l1 = torch.nn.L1Loss()
        r2metric = R2Score()
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]
                output = self.model(features)
                error_l1 = criterion_l1(output, labels)
                errors.update(error_l1.item(), bsz)
                r2metric.update(output, labels)

        if return_r2_score:
            return errors.avg, r2metric.compute()
        else:
            return errors.avg


    def run(self, parser):
        self.set_up_training(parser)
        save_file_best = os.path.join(self.opt.save_folder, f"{self.opt.model_name}_best.pth")
        save_file_last = os.path.join(self.opt.save_folder, f"{self.opt.model_name}_last.pth")
        epoch_counter = range(self.start_epoch, self.opt.Regressor.trainer.epochs + 1)
        verbose = self.opt.Regressor.trainer.verbose
        if not verbose:
            epoch_counter = tqdm(epoch_counter)

        print("Start training regressor")
        for epoch in epoch_counter:
            self.adjust_learning_rate(self.opt.Regressor, epoch)
            self.train_epoch(epoch)
            valid_error = self.validation()
            if epoch == 1 or valid_error < best_error :
                best_error = valid_error
                self.save_model(epoch, save_file_best, others={"best_error": best_error}, verbose=verbose)

        # Save the last checkpoint
        self.save_model(epoch, save_file_last, others={"last_error": valid_error}, verbose=verbose)

        # Test the best model on the test set
        print("=" * 120)
        print("Test best model on test set...")
        checkpoint = torch.load(save_file_best)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
        test_error, test_r2_score = self.testing(return_r2_score=True)
        to_print = 'Test L1 error: {:.3f}, R2 score: {:.3f}'.format(test_error, test_r2_score)
        print(to_print)