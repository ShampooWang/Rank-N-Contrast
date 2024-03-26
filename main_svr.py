import sys
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from dataset import *
from utils import *
from AgeDB_exp import plot_max_dist

NAME2MODEL = {
    "linear": LinearRegression(),
    "svr": SVR(),
}

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=322, help="Random seed for reproducibility")
    parser.add_argument('--embed_dim', type=int, default=2, help="Dimension of embeddings")
    parser.add_argument('--data_csv', type=str, default='./data/agedb.csv', help='Path to the csv file of data')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model parameters')
    parser.add_argument('--model_type', type=str, default="svr", choices=list(NAME2MODEL.keys()), help="Model types")

    return parser.parse_args()

def get_spherical_coordinates(dim):
    """
    https://en.wikipedia.org/wiki/N-sphere
    """
    phi = np.random.rand(dim-1) * np.array([np.pi] * (dim-2) + [2*np.pi])
    sin_phi = np.insert(np.sin(phi), 0, 1) # 1, sin(phi_1), ..., sin(phi_{n-1})
    cos_phi = np.insert(np.cos(phi), dim-1, 1) # cos(phi_1), cos(phi_2), ..., 1

    return np.cumprod(sin_phi) * cos_phi

def plot_2d_umap(feats, labels):
    print(feats.shape)

    if feats.shape[-1] == 2:
        embedding = feats
    else:
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(feats)

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='GnBu')
    plt.colorbar(scatter, label='Age')
    plt.title('2D UMAP colored by Age')
    plt.savefig("./pics/AgeDB_concentric_x.png")

def get_age2feats(feats, labels):
    age2feats = defaultdict(list)
    for x, y in zip(feats, labels):
        age2feats[y].append(x)
    for age in age2feats:
        age2feats[age] = np.stack(age2feats[age], axis=0)
    age2feats = dict(sorted(age2feats.items())) # Sort age2feats by the age

    return age2feats

def getData(args):
    assert os.path.exists(args.data_csv)
    df = pd.read_csv(args.data_csv)
    sphere_coord = get_spherical_coordinates(args.embed_dim)
    get_concentric_x = lambda x: np.array(x["age"])[:, None] * sphere_coord[None, :]
    data = { split: {"X": get_concentric_x(df[df['split'] == split]), "y": np.array(df[df['split'] == split]["age"])} for split in ["train", "val", "test"] }

    return data

def train(args, X_train, y_train):
    reg = NAME2MODEL[args.model_type]
    reg.fit(X_train, y_train)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(reg, f)
            f.close()

    return reg

def test(model, X_test, y_test):
    X_pred = model.predict(X_test)
    print(X_pred)
    print(y_test)
    
    return np.mean(np.absolute(X_pred - y_test))

def main():
    args = parseArgs()
    np.random.seed(args.seed)
    data = getData(args)

    # plot_2d_umap(data["test"]["X"], data["test"]["y"])
    # plot_max_dist(get_age2feats(data["test"]["X"], data["test"]["y"]), group_by_age=True)

    reg = train(args, data["train"]["X"], data["train"]["y"])
    mae = test(reg, data["test"]["X"], data["test"]["y"])

    print(mae)

if __name__ == "__main__":
    main()
