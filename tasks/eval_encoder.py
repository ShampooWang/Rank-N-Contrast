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
import matplotlib.pyplot as plt


class EvalEncoder(BaseTask):
    def __init__(self):
        super().__init__()

    def add_general_arguments(self, parser):
        parser = super().add_general_arguments(parser)
        parser.add_argument('--sup_resnet', action='store_true', help='whether to use supervised ResNet')
        parser.add_argument('--add_sup_resnet', type=str, default=None, help='whether to add supervised ResNet')
        parser.add_argument('--regerssor_ckpt', type=str, default=None, help='path to the trained regressor')
        parser.add_argument('--umap_pic_name', type=str, default=None, help='Name of the umap picture')
        parser.add_argument('--error_pic_name', type=str, default=None, help='Name of the error picture')
        parser.add_argument('--ydiff_pic_name', type=str, default=None, help='Name of the ydiff picture')
        parser.add_argument('--k', type=int, default=5)

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

        # Add supervised resnet for computing mae gains
        if self.opt.add_sup_resnet is not None:
            assert os.path.exists(self.opt.add_sup_resnet), self.opt.add_sup_resnet
            print(f"Add referenced supervised resnet for computing MAE gains from {self.opt.add_sup_resnet}")
            supResNet = SupResNet(name=self.opt.Encoder.type, num_classes=get_label_dim(self.opt.data.dataset))
            supResNet.load_state_dict(torch.load(self.opt.add_sup_resnet, map_location='cpu')["model"])
            self.supResNet = supResNet.cuda()
            self.label2error_supResNet = defaultdict(list) # Collecting testing errors of the reference supresnet
        else:
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
            for data_dict in tqdm(self.train_loader):
                images = data_dict["img"].cuda(non_blocking=True)
                labels = data_dict["label"].cuda(non_blocking=True)
                bsz = labels.shape[0]

                output = None
                if self.opt.sup_resnet:
                    features = self.model.extract_features(images)
                    output = self.model.fc(features)
                else:
                    features = self.model(images)
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
        data_occurences = self.train_dataset.get_data_occurences()

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
                data_dist=data_occurences,
                label2error=label2error,
                pic_name=self.opt.error_pic_name,
                ref_errors=getattr(self, "label2error_supResNet", None)
            )

        label2error_reduced = { int(lab): sum(errors)/len(errors) for lab, errors in label2error.items() if len(errors) > 0 }
        label2error_reduced = dict(sorted(label2error_reduced.items()))
        # max_error_lab = max(label2error_reduced, key=label2error_reduced.get)
        # print(max_error_lab)
        # print(label2error[max_error_lab])
        # max_error_feats = age2feats[max_error_lab]
        # pairwise_dists = np.linalg.norm(max_error_feats[:, None, :] - Z[None, :, :], axis=-1)
        # pairwise_dists[pairwise_dists == 0] = 1
        # print((np.abs(max_error_lab - y)[None, :] / pairwise_dists).mean(1))
        # print(f"Max data occurences: {max(data_occurences, key=data_occurences.get)}")
        # plot_correlation_distribution(Z, y, data_occurences, pic_name="e2e_l1")

        # plt.imshow(pairwise_dists, cmap="Oranges")
        # plt.colorbar()
        # plt.gca().set_yticks([])  # This removes the y-axis ticks
        # plt.grid(False)  # Ensure no other grid lines interfere
        # plt.savefig("test.png")

        # if hasattr(self, "label2error_supResNet"):
        #     uniform_data_num = 100 # 2 * round(len(self.train_dataset) / 102)
        #     print(uniform_data_num)
        #     zero_shot = {}
        #     many_shot = {}
        #     few_shot = {}
        #     for age in label2error:
        #         mae_gains = (np.array(label2error[age]) - np.array(self.label2error_supResNet[age])).sum()
        #         if age not in data_occurences:
        #             zero_shot[age] = mae_gains
        #         elif data_occurences[age] >= uniform_data_num:
        #             many_shot[age] = mae_gains
        #         else:
        #             few_shot[age] = mae_gains
            
        #     def compute_improve_percent(score_dict: dict):
        #         if len(score_dict) == 0: return 0
        #         total_num = 0
        #         for score in score_dict.values():
        #             if score < 0:
        #                 total_num += 1
        #         return total_num / len(score_dict)
            
        #     print(f"Many shot total mae gains: {sum(many_shot.values())}")
        #     print(f"Many shot mae improve percent: {compute_improve_percent(many_shot):.3f}")
        #     print(f"Few shot total mae gains: {sum(few_shot.values())}")
        #     print(f"Few shot mae improve percent: {compute_improve_percent(few_shot):.3f}")
        #     print(f"Zero shot total mae gains: {sum(zero_shot.values())}")

        age2nbr_ydiffs = plot_knbr_ydiffs_distribution(Z, y, data_occurences, pic_name=self.opt.ydiff_pic_name, k=self.opt.k + 1)
        print(np.corrcoef(list(age2nbr_ydiffs.values()), list(label2error_reduced.values()))[0,1])

        # avg_y_diffs_reduced = plot_std_ydiffs_distribution(Z, y, data_occurences, "e2e_l1")
        # print(np.corrcoef(list(avg_y_diffs_reduced.values()), list(label2error_reduced.values()))[0,1])