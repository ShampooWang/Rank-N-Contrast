import torch
import torch.utils
import torch.utils.data
import datasets
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

def main():
    data = pd.read_csv("/tmp2/jeffwang/Rank-N-Contrast/datasets/AgeDB/agedb_rank.csv")
    data_folder = "/tmp2/jeffwang/Rank-N-Contrast/datasets/AgeDB"

    # Check grey scale
    with open("grey_scale.csv", "w") as f:
        f.write("age,path,split,greyscale\n")
        for i in tqdm(range(len(data))):
            row = data.iloc[i]
            img = Image.open(os.path.join(data_folder, row["path"]))
            color_count = img.getcolors()
            if color_count: 
                # your image is grayscale
                f.write(f"{row['age']},{row['path']},{row['split']},1\n")
            else:
                # your images is colored
                f.write(f"{row['age']},{row['path']},{row['split']},0\n")

    new_data = pd.read_csv("grey_scale.csv")
    print(new_data)


if __name__ == "__main__":
    main()