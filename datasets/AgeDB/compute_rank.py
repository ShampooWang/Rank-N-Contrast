import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

def main():
    # Load the data
    data = pd.read_csv('agedb.csv')
    train_data = data[data["split"] != "test"]
    y_ranges = set(train_data["age"])
    y_to_rank = { y: rank for rank, y in enumerate(y_ranges) }
    with open("agedb_rank.csv", "w") as f:
        f.write("age,path,split,rank\n")
        for i in range(len(data)):
            row = data.iloc[i]
            rank = y_to_rank[row['age']] if row['split'] != "test" else -1
            f.write(f"{row['age']},{row['path']},{row['split']},{rank}\n")
    rank_data = pd.read_csv('agedb_rank.csv')
    print(rank_data)
    


if __name__ == "__main__":
    main()