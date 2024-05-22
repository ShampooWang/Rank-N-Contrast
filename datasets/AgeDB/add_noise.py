import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

def main(scale):
    print(scale)

    # Load the data
    data = pd.read_csv('agedb.csv')

    # Select the training data
    is_train = data["split"] == "train"
    train_ages = data.loc[is_train, "age"].astype(np.float32)

    # Add Gaussian noise to the training ages
    noise = scale * np.random.normal(size=train_ages.shape)
    train_ages += noise

    # Update the ages in the DataFrame
    data.loc[is_train, "age"] = train_ages
    print(data)
    
    data.to_csv(f"agedb_noise_scale_{scale}_new.csv")


if __name__ == "__main__":
    np.random.seed(322)
    for scale in [1.0, 2.0, 5.0]:
        main(scale=scale)