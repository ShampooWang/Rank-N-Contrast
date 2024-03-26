import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import umap

from main_linear import parse_option, set_loader, set_model
from utils import *

def get_test_loss(test_loader, model, regressor):
    model.eval()
    regressor.eval()

    losses = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            features = model(images)
            output = regressor(features)

            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)

    return losses

def get_feats_grouped_by_age(test_loader, model, regressor):
    model.eval()
    regressor.eval()

    age2feats = defaultdict(list)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            features = model(images)
            # output = regressor(features)
            
            for lab, feat in zip(labels, features):
                age2feats[lab.item()].append(feat)

    for age in age2feats:
        age2feats[age] = torch.stack(age2feats[age], dim=0).cpu().numpy()

    return age2feats

def plot_2d_umap(feats, labels):
    print(feats.shape)
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(feats)

    # Step 4: Plotting
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='GnBu')
    plt.colorbar(scatter, label='Age')
    plt.title('2D UMAP colored by Age')
    plt.savefig("./pics/AgeDB.png")

def plot_max_dist(age2feats, show_value=False, group_by_age=False):
    dist_mtx = np.zeros([len(age2feats)]*2)
    start_idices = np.cumsum([0] + [ age2feats[age].shape[0] for age in age2feats])
    all_feats = np.concatenate([ feats for feats in age2feats.values() ], axis=0) # num x 512
    all_dists = np.linalg.norm((all_feats[None, :, :] - all_feats[:, None, :]), axis=-1)
    # all_dists = squareform(pdist(all_feats))

    if group_by_age:
        for i, (rs, re) in enumerate(zip(start_idices[:-1], start_idices[1:])):
            for j, (cs, ce) in enumerate(zip(start_idices[i:-1], start_idices[1+i:])):
                dist_mtx[i, i+j] = dist_mtx[i+j, i] = np.max(all_dists[rs:re, cs:ce])
    else:
        dist_mtx = all_dists

    # Plot using imshow with a blue colormap
    plt.figure(figsize=(60, 40) if show_value else (12,8))
    plt.imshow(dist_mtx, cmap='Oranges')
    plt.colorbar()  # Show color scale

    # Loop over data dimensions and create text annotations.
    if show_value:
        for i in range(dist_mtx.shape[0]):
            for j in range(dist_mtx.shape[1]):
                text = plt.text(j, i, np.round(dist_mtx[i, j], 1),
                            ha="center", va="center", color="black")
                
    if group_by_age:
        plt.xlabel("Age")
        plt.ylabel("Age")
    else:
        plt.xlabel("Data index")
        plt.ylabel("Data index")

    plt.savefig('./pics/dist_group_false.png')       


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, regressor, criterion = set_model(opt)
    checkpoint = torch.load("./AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/Regressor_AgeDB_ep_100_lr_0.05_d_0.2_wd_0_mmt_0.9_bsz_64_trial_0_best.pth")
    regressor.load_state_dict(checkpoint['state_dict'])

    
    age2feats = get_feats_grouped_by_age(test_loader, model, regressor)
    age2feats = dict(sorted(age2feats.items())) # Sort age2feats by the age

    # Plot 2d umap
    plot_2d_umap(feats=np.concatenate([ feats for feats in age2feats.values() ], axis=0),
                 labels=sum([[age] * age2feats[age].shape[0] for age in age2feats], [])
    )
        
    # Compute maximum distance for each age with other ages
    # plot_max_dist(age2feats)


if __name__ == "__main__":
    main()