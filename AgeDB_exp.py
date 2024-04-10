import logging
import argparse
import sys
import torch
import numpy as np
from numpy.linalg import matrix_rank
import os

from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from model import Encoder, model_dict, SupResNet
from main_linear import set_loader, set_model
from utils import *
from loss import PointwiseRankingLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
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

    # supervised resent
    parser.add_argument('--sup_resnet', type=bool, default=False, help='whether to use supervised ResNet')

    # regressor
    parser.add_argument('--add_regressor', type=bool, default=False, help='whether to add linear regressor module')
    parser.add_argument('--regerssor_ckpt', type=str, default=None, help='path to the trained regressor')

    # others
    parser.add_argument('--sph', type=bool, default=False, help='Using spherical coordinates to regularize the features')
    parser.add_argument('--umap_pic_name', type=str, default='', help='Name of the umap picture')
    

    opt = parser.parse_args()

    opt.model_name = 'Regressor_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.epochs, opt.learning_rate, opt.lr_decay_rate,
               opt.weight_decay, opt.momentum, opt.batch_size, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1])

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

def get_test_loss(test_loader, model, opt, model_opt, regressor, pretrain_criterion=None, extract_features=True):
    model.eval()
    if regressor is not None:
        regressor.eval()
    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()
    age2feats = defaultdict(list)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            print(images.shape)
            exit(0)
            if opt.sup_resnet:
                features = model.extract_features(images)
                output = model.fc(features)
            else:
                features = model.encoder(images)
                if regressor is not None:
                    output = regressor(features)
                else:
                    if pretrain_criterion is not None and hasattr(pretrain_criterion, "anchor"):
                        scores = torch.matmul(features.double(), pretrain_criterion.anchor.unsqueeze(1)) # size [2B, 1]
                        features = features * scores / features.norm(dim=-1, p=2).unsqueeze(1)

                    assert hasattr(model_opt, 'feature_norm')
                    if model_opt.feature_norm == "l1":
                        output = features.norm(dim=-1, p=1).unsqueeze(1)
                    elif model_opt.feature_norm == "l2":
                        output = features.norm(dim=-1, p=2).unsqueeze(1)
                    else:
                        raise NotImplementedError(model_opt.feature_norm)
                
            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

            if extract_features:
                for lab, feat in zip(labels, features):
                    age2feats[lab.item()].append(feat)

    if extract_features and len(age2feats) > 0:
        for age in age2feats:
            age2feats[age] = torch.stack(age2feats[age], dim=0).cpu().numpy()

    return losses, age2feats    

def set_model(opt):
    model = Encoder(name=opt.model)
    criterion = torch.nn.L1Loss()

    dim_in = model_dict[opt.model][1]
    dim_out = get_label_dim(opt.dataset)
    
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    model_opt = ckpt['opt'] if 'opt' in ckpt else None

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model = model.cuda()
    
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict)
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    if opt.add_regressor:
        regressor = torch.nn.Linear(dim_in, dim_out)
        regressor = regressor.cuda()
    else:
        regressor = None

    return model, model_opt, regressor, criterion

def set_supResBet(opt):
    model = SupResNet(name=opt.model, num_classes=get_label_dim(opt.dataset))
    criterion = torch.nn.L1Loss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True
        state_dict = torch.load(opt.ckpt, map_location='cpu')
        model.load_state_dict(state_dict['model'])

    return model, None, criterion, None

def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    set_model_func = set_supResBet if opt.sup_resnet else set_model
    model, model_opt, regressor, criterion = set_model_func(opt)
    if regressor is not None and opt.regerssor_ckpt is not None:
        print(f"loading regressor checkpoint from {opt.regerssor_ckpt}")
        checkpoint = torch.load(opt.regerssor_ckpt)
        regressor.load_state_dict(checkpoint['state_dict'])

    # pretrain_criterion = PointwiseRankingLoss(objective="ordinal")
    # loss_params = torch.load(opt.ckpt)['criterion']
    # pretrain_criterion.load_state_dict(loss_params)
    # pretrain_criterion.cuda()

    test_loss, age2feats = get_test_loss(test_loader, model, opt, model_opt, regressor, pretrain_criterion=None)   
    if len(age2feats) > 0: 
        age2feats = dict(sorted(age2feats.items())) # Sort age2feats by the age
    feats = np.concatenate([ feats for feats in age2feats.values() ], axis=0) 
    print(f"Test loss: {test_loss.avg}")

    # Plot 2d umap
    if opt.sph:
        sph_coor = np.stack([ get_spherical_coordinates(512, seed=i) for i in range(feats.shape[0]) ], axis=0)
        feats = np.linalg.norm(feats, axis=1)[:, None] * sph_coor
    plot_2d_umap(
        feats=feats,
        labels=sum([[age] * age2feats[age].shape[0] for age in age2feats], []),
        save_path=f'./pics/2d_umap/{opt.dataset}_{opt.umap_pic_name}.png'
    )
        
    # Compute svd
    print(matrix_rank(feats))
    # plot_svd(feats, './pics/pwr_corr_atanh_svd.png')


if __name__ == "__main__":
    main()