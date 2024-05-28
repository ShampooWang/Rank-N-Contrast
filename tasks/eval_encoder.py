import argparse
import os
import sys
import logging
import torch
from torcheval.metrics import R2Score
from collections import defaultdict
from model import Encoder, model_dict, SupResNet
from utils import *
from .base_task import BaseTask
# Data
import datasets
from torch.utils import data
# Losses
from loss import *
# Others
from exp_utils import *
from tqdm import tqdm


class EvalEncoder(BaseTask):
    def __init__(self):
        super().__init__()

    def add_general_arguments(self, parser):
        parser = super().add_general_arguments(parser)
        parser.add_argument('--sup_resnet', action='store_true', help='whether to use supervised ResNet')
        parser.add_argument('--add_sup_resnet', type=str, default=None, help='whether to add supervised ResNet')
        parser.add_argument('--regerssor_ckpt', type=str, default=None, help='path to the trained regressor')
        parser.add_argument('--umap_pic_name', type=str, default=None, help='Name of the umap picture')
        parser.add_argument('--error_pic_name', type=str, default=None, help='Name of the umap picture')

        return parser

    def set_model_and_criterion(self):
        ckpt = torch.load(self.opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # Load model
        if self.opt.sup_resnet:
            model = SupResNet(name=self.opt.Encoder.type, num_classes=get_label_dim(self.opt.data.dataset))
        else:
            model = Encoder(name=self.opt.Encoder.type)
            state_dict = { k.replace("module.", ""): v for k, v in state_dict.items() }
        model = model.cuda()
        model.load_state_dict(state_dict)
        print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {self.opt.ckpt}!")
        self.model = model

        # Load regressor
        if self.opt.regerssor_ckpt is not None:
            dim_in = model_dict[self.opt.Encoder.type][1]
            dim_out = get_label_dim(self.opt.data.dataset)
            regressor = torch.nn.Linear(dim_in, dim_out).cuda()
            reg_ckpt = torch.load(self.opt.regerssor_ckpt)
            if "model" in reg_ckpt:
                regressor.load_state_dict(reg_ckpt["model"])
            else:
                regressor.load_state_dict(reg_ckpt["state_dict"])
            print(f"<=== Epoch [{ckpt['epoch']}] regressor checkpoint Loaded from {self.opt.regerssor_ckpt}!")
            self.regressor = regressor

        # Add supervised resnet for computing mae gains
        if self.opt.add_sup_resnet is not None:
            assert os.path.exists(self.opt.add_sup_resnet), self.opt.add_sup_resnet
            print(f"Add referenced supervised resnet for computing MAE gains from {self.opt.add_sup_resnet}")
            supResNet = SupResNet(name=self.opt.Encoder.type, num_classes=get_label_dim(self.opt.data.dataset))
            supResNet.load_state_dict(torch.load(self.opt.add_sup_resnet, map_location='cpu')["model"])
            self.supResNet = supResNet.cuda()
            self.label2error_supResNet = defaultdict(list) # Collecting testing errors of the reference supresnet

        # Set criterion
        # self.criterion = torch.nn.L1Loss().cuda()

    def set_loader(self):
        dataset = datasets.__dict__[self.opt.data.dataset]
        self.train_dataset = dataset(seed=self.opt.seed, split='train', **self.opt.data)
        self.val_dataset = dataset(seed=self.opt.seed, split='val', **self.opt.data)
        self.test_dataset = dataset(seed=self.opt.seed, split='test', **self.opt.data)

        print(f'Train set size: {self.train_dataset.__len__()}')
        print(f'Valid set size: {self.val_dataset.__len__()}')
        print(f'Test set size: {self.test_dataset.__len__()}')

        batch_size = 64
        num_workers = 4
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
        )
        
    def testing(self):
        self.model.eval()
        if hasattr(self, "regressor"):
            self.regressor.eval()
        
        if hasattr(self, "supResNet"):
            self.supResNet.eval()

        losses_mae = AverageMeter()
        losses_se = AverageMeter()
        criterion_l1 = torch.nn.L1Loss(reduction='none').cuda()
        criterion_l2 = torch.nn.MSELoss().cuda()
        age2feats = defaultdict(list)
        label2error = defaultdict(list)
        r2metric = R2Score()

        with torch.no_grad():
            for data_dict in tqdm(self.val_loader):
                images = data_dict["img"].cuda(non_blocking=True)
                labels = data_dict["label"].cuda(non_blocking=True)
                bsz = labels.shape[0]

                output = None
                if self.opt.sup_resnet:
                    features = self.model.extract_features(images)
                    output = self.model.fc(features)
                else:
                    features = self.model.encoder(images)
                    if hasattr(self, "regressor"):
                        output = self.regressor(features)

                if output is not None:
                    loss_l1 = criterion_l1(output, labels)
                    loss_l2 = criterion_l2(output, labels)
                    losses_mae.update(loss_l1.mean().item(), bsz)
                    losses_se.update(loss_l2.item(), bsz)
                    r2metric.update(output, labels)

                    for lab, error in zip(labels, loss_l1):
                        label2error[lab.item()].append(error.item())

                    if hasattr(self, "supResNet") and hasattr(self, "label2error_supResNet"):
                        for lab, error in zip(labels, (self.supResNet(images) - labels).abs()):
                            self.label2error_supResNet[lab.item()].append(error.item())

                for lab, feat in zip(labels, features):
                    age2feats[lab.item()].append(feat)

                del features, images, labels
                    
        if len(age2feats) > 0:
            for age in age2feats:
                age2feats[age] = torch.stack(age2feats[age], dim=0).cpu().numpy()

        return losses_mae, losses_se, age2feats, label2error, r2metric.compute()


    def run(self, parser):
        self.parse_option(parser, log_file=False)
        self.set_seed()
        self.set_model_and_criterion()
        self.set_loader()

        # Testing
        print("Start testing")
        losses_mae, losses_se, age2feats, label2error, R2score = self.testing()

        if losses_mae.count > 0:
            print(f"Test MAE error: {losses_mae.avg:.3f}")
            print(f"Test R2 score: {R2score:.3f}")

        if losses_se.count > 0:
            print(f"Test RMSE error: {math.sqrt(losses_se.avg):.3f}")

        # Sort age2feats by the age
        if len(age2feats) > 0: 
            age2feats = dict(sorted(age2feats.items())) 
        Z = np.concatenate([ feats for feats in age2feats.values() ], axis=0) 
        y = np.array(sum([[age] * age2feats[age].shape[0] for age in age2feats], []))


        # Plot 2d umap
        if self.opt.umap_pic_name is not None:
            plot_2d_umap(
                feats=Z,
                labels=y,
                save_path=f'./pics/2d_umap/{self.opt.umap_pic_name}.png'
            )

        # Plot error distribution
        if self.opt.error_pic_name is not None:
            plot_error_distribution(
                data_dist=self.train_dataset.get_data_occurences(),
                label2error=label2error,
                pic_name=self.opt.error_pic_name,
                ref_errors=getattr(self, "label2error_supResNet", None)
            )