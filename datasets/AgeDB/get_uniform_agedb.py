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
   data_occ = get_distribution(data)
   uniform_data_num = round(len(data) / 102)
   data_split_unit = round(uniform_data_num / 8)
   print(uniform_data_num)

   new_data = []
   new_data_occ = np.zeros(102)
   for i in range(len(data)):
      row = data.iloc[i]
      age = int(row["age"])
      if data_occ[age] > uniform_data_num and new_data_occ[age] < uniform_data_num:
         new_data_occ[age] += 1

         if new_data_occ[age] < 6 * data_split_unit:
            split = "train"
         elif 6 * data_split_unit <= new_data_occ[age] < 7 * data_split_unit:
            split = "val"
         else:
            split = "test"

         new_data.append({
            "age": age,
            "path": row["path"],
            "split": split,
         })

   with open("uniform_agedb.csv", "w") as f:
      f.write("age,path,split\n")
      for i in tqdm(range(len(new_data))):
         row = new_data[i]
         f.write(f"{row['age']},{row['path']},{row['split']}\n")
      f.close()

   print(pd.read_csv("uniform_agedb.csv"))
   
if __name__ == "__main__":
    main()