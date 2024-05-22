import os
import argparse
import sys
import logging
import torch
from torcheval.metrics import R2Score
import time
from model import SupResNet
from utils import *
from .base_task import BaseTask


print = logging.info


class TrainE2EEncoder(BaseTask):
    def __init__(self):
        super().__init__()

    def add_general_arguments(self, parser):
        parser = super().add_general_arguments(parser)
        parser.add_argument("--two_view_aug", action="store_false", help="Add two views of augmentation for training")

        return parser

    def create_modelName_and_saveFolder(self, opt):
        opt.model_path = './checkpoints/L1'
        opt.model_name = f'{opt.data.dataset}_{opt.Encoder.type}_ep_{opt.Encoder.trainer.epochs}'
        opt.model_name += f"2view_{opt.two_view_aug}"

        if getattr(opt.data, "noise_scale", 0.0) > 0.0:
            opt.model_name += f"_noise_scale_{opt.data.noise_scale}"

        if opt.fix_model_and_aug:
            opt.model_name += "fix_model&aug"

        opt.model_name += f"_seed_{opt.seed}"
        opt.model_name += f"_trial_{opt.trial}"

        if opt.resume is not None:
            opt.model_name = opt.resume.split('/')[-2]

        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        else:
            print('WARNING: folder exist.')

        print(f"Model name: {opt.model_name}")
        print(f"Options: {opt}")

        self.log_file_path = os.path.join(opt.save_folder, 'training.log')

        return opt

    def set_model_and_criterion(self):
        self.model = SupResNet(name=self.opt.Encoder.type, num_classes=get_label_dim(self.opt.data.dataset))
        self.criterion = torch.nn.L1Loss()

        if self.opt.fix_model_and_aug:
            model_path = f"./checkpoints/seed322/sup{self.opt.Encoder.type}r.pth"
            model_params = torch.load(model_path)["model"]
            self.model.load_state_dict(model_params)
            print(f"Fixing e2e encoder's initialization. Model checkpoint Loaded from {model_path}!")

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model.encoder = torch.nn.DataParallel(self.model.encoder)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def set_loader(self):
        batch_size = self.opt.Encoder.trainer.batch_size
        num_workers = self.opt.Encoder.trainer.num_workers
        super().set_loader(batch_size, num_workers, self.opt.two_view_aug)

    def train_epoch(self, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for idx, data_dict in enumerate(self.train_loader):
            images, labels = data_dict["img"], data_dict["label"]
            data_time.update(time.time() - end)
            bsz = labels.shape[0]

            # Two crop transform
            if isinstance(images, list):
                images = torch.cat([images[0], images[1]], dim=0)
                labels = labels.repeat(2, 1)

            if torch.cuda.is_available():
                labels = labels.cuda(non_blocking=True)
                images = images.cuda(non_blocking=True)

            bsz = labels.shape[0]
            output = self.model(images)
            loss = self.criterion(output, labels)
            losses.update(loss.item(), bsz)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % self.opt.Encoder.trainer.print_freq == 0:
                to_print = 'Train: [{0}][{1}/{2}]\t' \
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch, idx + 1, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses)
                print(to_print)
                sys.stdout.flush()

        
    def validation(self, return_r2_score=False):
        self.model.eval()
        errors = AverageMeter()
        r2metric = R2Score()

        with torch.no_grad():
            for data_dict in self.val_loader:
                images = data_dict["img"].cuda(non_blocking=True)
                labels = data_dict["label"].cuda(non_blocking=True)
                bsz = labels.shape[0]
                output = self.model(images)
                error = self.criterion(output, labels)
                errors.update(error.item(), bsz)
                r2metric.update(output, labels)

        if return_r2_score:
            return errors.avg, r2metric.compute()
        else:
            return errors.avg

    def testing(self, return_r2_score=False):
        self.model.eval()
        errors = AverageMeter()
        r2metric = R2Score()

        with torch.no_grad():
            for data_dict in self.test_loader:
                images = data_dict["img"].cuda(non_blocking=True)
                labels = data_dict["label"].cuda(non_blocking=True)
                bsz = labels.shape[0]
                output = self.model(images)
                error = self.criterion(output, labels)
                errors.update(error.item(), bsz)
                r2metric.update(output, labels)

        if return_r2_score:
            return errors.avg, r2metric.compute()
        else:
            return errors.avg

    def run(self, parser):
        self.set_up_training(parser)
        save_file_best = os.path.join(self.opt.save_folder, 'best.pth')

        # training routine
        for epoch in range(self.start_epoch, self.opt.Encoder.trainer.epochs + 1):
            self.adjust_learning_rate(self.opt.Encoder, epoch)
            self.train_epoch(epoch)
            val_error = self.validation()

            if epoch == 1 or val_error < best_error:
                best_error = val_error
                self.save_model(epoch, save_file_best, others={"best_error": best_error})

            print('Val L1 error: {:.3f}'.format(val_error))
            print(f"Best Error: {best_error:.3f}")

            if epoch % self.opt.Encoder.trainer.save_freq == 0:
                save_epoch_file = os.path.join(self.opt.save_folder, f'ckpt_epoch_{epoch}.pth')
                self.save_model(epoch, save_epoch_file)

            if epoch % self.opt.Encoder.trainer.save_curr_freq == 0:
                save_curr_file = os.path.join(self.opt.save_folder, "curr_last.pth")
                self.save_model(epoch, save_curr_file)

        print("=" * 120)
        print("Test best model on test set...")
        checkpoint = torch.load(save_file_best)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
        test_loss, test_r2_score = self.testing(return_r2_score=True)
        to_print = 'Test L1 error: {:.3f}, R2 score: {:.3f}'.format(test_loss, test_r2_score)
        print(to_print)