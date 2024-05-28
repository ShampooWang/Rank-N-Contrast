import os
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_2d_umap(feats, labels, save_path='./pics/AgeDB.png'):
    print(feats.shape)
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(feats)

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='GnBu')
    plt.colorbar(scatter, label='Age')
    plt.title('2D UMAP colored by Age')
    plt.savefig(save_path)


def plot_max_dist(age2feats, show_value=False, group_by_age=False, save_path='./pics/dist_group_false.png'):
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

    plt.savefig(save_path)  

def plot_svd(feats, save_path):
    C = np.cov(feats.T, bias=True)
    U, s, Vh = np.linalg.svd(C)
    plt.plot(np.log10(s))
    plt.xlabel('Singular Value Rank Index')
    plt.ylabel('Log of singular values')
    plt.savefig(save_path)

def plot_error_distribution(data_dist, label2error, pic_name, pic_directory="./pics/error_distribution/", ref_errors=None):
    ages = np.arange(0, 102)  # Ages from 0 to 101
    if isinstance(label2error[0], list):
        label2error = { int(lab): sum(errors)/len(errors) for lab, errors in label2error.items() if len(errors) > 0 }
    label2error = dict(sorted(label2error.items()))

    # Create num_samples and mae_dist (or mae_gains)
    num_samples = np.zeros(102)
    num_samples[list(data_dist.keys())] += np.array(list(data_dist.values()))
    if ref_errors is not None:
        if isinstance(ref_errors[0], list):
            ref_errors = { int(lab): sum(errors)/len(errors) for lab, errors in ref_errors.items() if len(errors) > 0 }
        ref_errors = dict(sorted(ref_errors.items()))
        mae_gains = np.zeros(102)
        mae_gains[list(label2error.keys())] += np.array(list(label2error.values())) - np.array(list(ref_errors.values()))
    else:
        mae_dist = np.zeros(102)
        mae_dist[list(label2error.keys())] += np.array(list(label2error.values()))

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot label distribution
    axs[0].bar(ages, num_samples, color='blue')
    axs[0].set_ylabel('# of training samples')
    axs[0].set_title('Label Distribution and MAE by Age')

    # Plot MAE gains
    axs[1].set_xlabel('Target value (Age)')
    if ref_errors is not None:
        axs[1].bar(ages, mae_gains, color='orange')
        axs[1].set_ylabel('MAE gains')
        axs[1].axhline(0, color='gray', linewidth=0.8)  # Zero line for reference
        pic_name += "_mae_gains"
    else:
        axs[1].bar(ages, mae_dist, color='orange')
        axs[1].set_ylabel('MAE')
    

    if not os.path.exists(pic_directory):
        os.makedirs(pic_directory)

    plt.savefig(os.path.join(pic_directory, f"{pic_name}.png"))


def plot_feature_norm_distribution(age2feats, save_path):
    age2feats_norm = { lab: np.linalg.norm(feats, axis=1).mean(0) for lab, feats in age2feats.items() }
    labels = [ f"{lab}" for lab in age2feats.keys() ]
    norms = list(age2feats_norm.values())
    print(np.corrcoef(list(age2feats_norm.keys()), norms))
    
    assert len(labels) == len(norms)
    x = [ i+1 for i in range(len(labels)) ]
    fig, ax = plt.subplots(figsize = (15,4))
    ax.tick_params(axis='x', labelsize=5)
    ax.bar(x, norms)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    plt.title('Age vs. Feature norm')
    plt.xlabel('Age')
    plt.ylabel('Feature norm')
    plt.savefig(save_path)

def compute_norm_confusion_matrix(label2feats: dict, save_dir: str, img_name: str, show_value=True):
    os.makedirs(save_dir, exist_ok=True)
    min_norms = { lab: np.linalg.norm(feats, axis=1).min(0) for lab, feats in label2feats.items() }
    max_norms = { lab: np.linalg.norm(feats, axis=1).max(0) for lab, feats in label2feats.items() }
    
    label_num = len(label2feats)
    confusion_matrix = np.zeros([label_num, label_num])
    for i, (label, feats) in enumerate(label2feats.items()):
        feats_norm = np.linalg.norm(feats, axis=1, keepdims=True) # size: N x 1
        other_min_norms = np.array([ error for (lab, error) in min_norms.items() ])
        other_max_norms = np.array([ error for (lab, error) in max_norms.items() ])
        is_in = (feats_norm >= other_min_norms[None, :]) *  (feats_norm <= other_max_norms[None, :]) # size: N x K-1
        confusion_matrix[i, :] = is_in.mean(0)

    # Plot using imshow with a blue colormap
    plt.figure(figsize=(60, 40) if show_value else (12,8))
    plt.imshow(confusion_matrix, cmap='Oranges')
    plt.colorbar()  # Show color scale

    # Loop over data dimensions and create text annotations.
    if show_value:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, np.round(confusion_matrix[i, j], 1), \
                        ha="center", va="center", color="black")
                
    # weight_mtx = np.zeros([label_num, label_num])
    # for i in range(1, label_num):
    #     weight_mtx += np.diag(np.full(label_num-i, i), i)
    #     weight_mtx += np.diag(np.full(label_num-i, i), -i)
    
    entropy = (confusion_matrix - np.eye(label_num)).sum()
    plt.title(f'Non-diagonal value sum: {entropy}')

    plt.savefig(os.path.join(save_dir, img_name))  

def get_spherical_coordinates(dim, seed=322):
    """
    https://en.wikipedia.org/wiki/N-sphere
    """
    np.random.seed(seed)
    phi = np.random.rand(dim-1) * np.array([np.pi] * (dim-2) + [2*np.pi])
    # phi = 0.322 * np.array([np.pi] * (dim-2) + [2*np.pi])
    sin_phi = np.insert(np.sin(phi), 0, 1) # 1, sin(phi_1), ..., sin(phi_{n-1})
    cos_phi = np.insert(np.cos(phi), dim-1, 1) # cos(phi_1), cos(phi_2), ..., 1

    return np.cumprod(sin_phi) * cos_phi

def check_delta_order(age2feats: dict):
    # Sort age2feats by the age
    age2feats = dict(sorted(age2feats.items())) 

    # Compute z_{i,j} and y_{i,j}
    z = np.concatenate([ feats for feats in age2feats.values() ], axis=0)  # [N, D]
    z_dists = np.linalg.norm((z[None, :] - z[:, None]), axis=-1) # [N, N]
    y = np.array(sum([[age] * age2feats[age].shape[0] for age in age2feats], [])) # [N]
    y_diffs = y[None, :] - y[:, None] # [N, N]

    # Remove diagonal
    N = z_dists.shape[0]
    diagonal_mask = (1 - np.eye(N)).astype(bool)
    z_dists = z_dists[diagonal_mask].reshape(N, N-1) # [N, N-1]
    y_abs_diffs = np.abs(y_diffs[diagonal_mask].reshape(N, N-1))
    z_dist_diffs = z_dists[:, None, :] - z_dists[:, :, None]
    del diagonal_mask

    # Compute violation
    result = 0.0 # result = {0.1: 0.0, 0.5: 0.0}
    for j in tqdm(range(N-1)):
        _y_abs_diffs = y_abs_diffs[:, j, None] # [N, 1]
        _dist_diffs = z_dist_diffs[:, j, :] # d_ik - d_ij, size: [N, N-1]
        _flipped_signs = np.sign(y_abs_diffs - y_abs_diffs[:, j, None]) # [N, N-1]

        eq_mask = np.equal(_y_abs_diffs, y_abs_diffs) # [N, N-1]
        eq_mask[:, j] = False
        neq_mask = ~eq_mask
        neq_mask[:, j] = False
        _flipped_z_dist_diffs = _dist_diffs * _flipped_signs

        # for delta in [0.1, 0.5]:
        # Check y_{i,j} == y_{i,k} parts
        # eq_violate_mask = np.abs(_dist_diffs) > delta
        # eq_violate_num = (eq_mask & eq_violate_mask).sum(1)

        # Check y_{i,j} != y_{i,k} parts    
        neq_violate_mask = _flipped_z_dist_diffs < 0 # neq_violate_mask = _flipped_z_dist_diffs < (1 / delta)
        neq_violate_num = (neq_mask & neq_violate_mask).sum(1) 

        _violation_percent = neq_violate_num / (N-1) # _violation_percent = (eq_violate_num + neq_violate_num) / (N-1)
        result += _violation_percent.mean(0) * 100 / (N-1) # result[delta] += _violation_percent.mean(0) * 100 / (N-1)

    print(result)

def compute_knn(Z, y, k: int=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='l2').fit(Z)
    distances, indices = nbrs.kneighbors(Z)
    nbrs_y = y[indices[:, 1:]] # [N, k]
    avg_nbrs_y_diffs = np.abs(y[:, None] - nbrs_y).mean()
    
    # print(f"{k} nearest neighbors label differences: {avg_nbrs_y_diffs:.3f}")

    return avg_nbrs_y_diffs

if __name__ == "__main__":
    from utils import set_seed
    set_seed(322)
    age2feats = { i: np.random.rand(50, 512) for i in range(100) }
    check_delta_order(age2feats=age2feats)
    
