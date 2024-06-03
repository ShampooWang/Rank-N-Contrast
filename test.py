import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def get_distribution(data: pd.DataFrame):
   labels, occurence = np.unique(data["age"], return_counts=True)
   return { int(age): occ for (age, occ) in zip(labels, occurence) }

def main():
   data = pd.read_csv("/tmp2/jeffwang/Rank-N-Contrast/datasets/AgeDB/agedb.csv")
   train_data = data[data["split"] == "train"]
   data_occ = get_distribution(train_data)
   total = 0
   for age, occ in data_occ.items():
      total += age * occ
   print(total / sum(data_occ.values()))
   
if __name__ == "__main__":
    main()