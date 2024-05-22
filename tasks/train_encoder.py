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
# Losses
from loss import *
# Others
from .train_regressor import TrainRegressor
from exp_utils import compute_knn
from tqdm import tqdm


print = logging.info


class TrainEncoder(BaseTask):
    def __init__(self):
        super().__init__()
        self.regressor_trainer = TrainRegressor()

    def add_general_arguments(self, parser):
        parser = super().add_general_arguments(parser)
        parser.add_argument("--two_view_aug", action="store_false", help="Add two views of augmentation for training")

        return parser
    
    def create_modelName_and_saveFolder(self, opt):
        loss_type = opt.Encoder.loss.loss_type 
        if "deltaorder" in loss_type:
            opt.model_name = '{}_{}_ep_{}_delta_{}_objective_{}'. \
                format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, opt.Encoder.loss.delta, opt.Encoder.loss.objective)
        elif loss_type == "RnC":
            opt.model_name = '{}_{}_ep_{}'. \
                format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs)
        elif loss_type == "pairwise":
            delta =  getattr(opt.Encoder.loss, "delta", 0.3)
            opt.model_name = '{}_{}_ep_{}_delta_{}_obj_{}'. \
                format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, delta, opt.Encoder.loss.objective) 
        elif loss_type  == "pointwise":
            opt.model_name = '{}_{}_ep_{}_norm_{}_obj_{}'. \
                format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, opt.Encoder.loss.feature_norm, opt.Encoder.loss.objective)  
        elif loss_type  == "ProbRank":
            t = getattr(opt.Encoder.loss, "t", 1.0)
            opt.model_name = '{}_{}_ep_{}_t_{}'. \
                format(opt.data.dataset, opt.Encoder.type, opt.Encoder.trainer.epochs, t)  
        else:
            raise NotImplementedError(loss_type)
        
        opt.model_name += f"_2view_{opt.two_view_aug}"

        if getattr(opt.data, "noise_scale", 0.0) > 0.0:
            opt.model_name += f"_noise_scale_{opt.data.noise_scale}"

        if opt.fix_model_and_aug:
            opt.model_name += "_fix_model&aug"

        opt.model_name += f"_seed_{opt.seed}"
        opt.model_name += f"_trial_{opt.trial}"
        
        if opt.resume is not None:
            opt.model_name = opt.resume.split('/')[-2]

        # Create folders
        opt.model_path = f'./checkpoints/{loss_type}'
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        else:
            print('WARNING: folder exist.')

        self.log_file_path = os.path.join(opt.save_folder, 'training.log')

        return opt

    def set_model_and_criterion(self):
        config = self.opt.Encoder.loss
        self.model = Encoder(name=self.opt.Encoder.type)
        if config.loss_type == "RnC":
            self.criterion = RnCLoss()
        elif config.loss_type == "pointwise":
            self.criterion = PointwiseRankingLoss(feature_norm=config.feature_norm, objective=config.objective)
        elif config.loss_type == "pairwise":
            self.criterion = PairwiseRankingLoss(delta=getattr(config, "delta", 0.3), objective=config.objective)
        elif config.loss_type == "ProbRank":
            self.criterion = ProbRankingLoss(t=getattr(config, "t", 1.0))
        else:
            self.criterion = DeltaOrderLoss(delta=config.delta, objective=config.objective)

        if self.opt.fix_model_and_aug:
            model_path = f"./checkpoints/seed322/{self.opt.Encoder.type}.pth"
            model_params = torch.load(model_path)["model"]
            self.model.load_state_dict(model_params)
            print(f"Fixing encoder's initialization. Model checkpoint Loaded from {model_path}!")

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
            ranks = data_dict["rank"].cuda(non_blocking=True) if "rank" in data_dict else None
            images, labels = data_dict["img"], data_dict["label"]
            data_time.update(time.time() - end)
            bsz = labels.shape[0]

            # Two crop transform
            if isinstance(images, list):
                images = torch.cat([images[0], images[1]], dim=0)
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                features = self.model(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            else:
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)                
                features = self.model(images)
                
            loss = self.criterion(features, labels, ranks)
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
                        'loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                        'feature_norm {avg_norms:.3f}'.format(
                    epoch, idx + 1, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, avg_norms=features.norm(dim=1).mean()
                )
                print(to_print)
                sys.stdout.flush()

        if hasattr(self.criterion, "weight"):
            print(f"Learnable weight: {self.criterion.weight.exp().item()}")

        if hasattr(self.criterion, "bias"):
            print(f"Learnable bias: {self.criterion.bias.exp().item()}")

        if hasattr(self.criterion, "t") and isinstance(self.criterion.t, torch.Tensor):
            print(f"Learnable t: {self.criterion.t.exp().item()}")

        
    def validation(self):
        self.model.eval()
        losses = AverageMeter()
        Z = []
        y = []
        with torch.no_grad():
            for data_dict in self.val_loader:
                images = data_dict["img"].cuda(non_blocking=True)
                labels = data_dict["label"].cuda(non_blocking=True)
                ranks = data_dict["rank"].cuda(non_blocking=True) if "rank" in data_dict else None
                features = self.model(images)
                bsz = labels.shape[0]

                loss = self.criterion(features, labels, ranks)
                losses.update(loss.item(), bsz)
                Z.append(features.cpu().numpy())
                y.append(data_dict["label"].numpy())

        Z = np.concatenate(Z, axis=0)
        y = np.concatenate(y, axis=0)
        avg_nbr_diffs = compute_knn(Z, y)

        print(f"Val loss: {losses.avg:.3f}\t5 nearest neighbors label differences: {avg_nbr_diffs:.3f}")

        return losses.avg, avg_nbr_diffs


    def run(self, parser):
        self.set_up_training(parser)
        self.regressor_trainer.parse_option(parser, log_file=False)
        best_val_loss_pth = os.path.join(self.opt.save_folder, 'best_val_loss.pth')
        best_nbr_ydiff_pth = os.path.join(self.opt.save_folder, 'best_nbr_ydiff.pth')
        curr_last_pth = os.path.join(self.opt.save_folder, 'curr_last.pth')

        # training routine
        for epoch in range(self.start_epoch, self.opt.Encoder.trainer.epochs + 1):
            self.adjust_learning_rate(self.opt.Encoder, epoch)
            self.train_epoch(epoch)
            val_loss, val_nbr_diffs = self.validation()

            if epoch == 1 or val_loss < best_loss:
                best_loss = val_loss
                self.save_model(epoch, best_val_loss_pth, others={"best_loss": best_loss})

            if epoch == 1 or val_nbr_diffs < best_nbr_ydiff:
                best_nbr_ydiff = val_nbr_diffs
                self.save_model(epoch, best_nbr_ydiff_pth, others={"best_nbr_ydiff": best_nbr_ydiff})

            if epoch % self.opt.Encoder.trainer.save_curr_freq == 0:
                self.save_model(epoch, curr_last_pth)

            if epoch > 1 and epoch != self.opt.Encoder.trainer.epochs and epoch % self.opt.Encoder.trainer.test_regression_freq == 0:
                self.regressor_trainer.opt.ckpt = curr_last_pth
                self.regressor_trainer.run(parser)

        # save and test the last model
        last_pth = os.path.join(self.opt.save_folder, 'last.pth')
        self.save_model(self.opt.Encoder.trainer.epochs, last_pth)
        self.regressor_trainer.opt.ckpt = last_pth
        self.regressor_trainer.run(parser)

        # best val loss
        print(f"Best pretraining validation loss: {best_loss:.3f}")
        self.regressor_trainer.opt.ckpt = best_val_loss_pth
        self.regressor_trainer.run(parser)

        # best avg_nbr_diffs
        print(f"Best avg_nbr_diffs: {best_nbr_ydiff:.3f}")
        self.regressor_trainer.opt.ckpt = best_nbr_ydiff_pth
        self.regressor_trainer.run(parser)