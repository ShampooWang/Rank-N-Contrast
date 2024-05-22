import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def main(split):

    # Load the data
    data = pd.read_csv('imdb_wiki.csv')
    target_data = data[data["split"] == split]
    distribution = defaultdict(int)

    for i in tqdm(range(len(target_data))):
        if i > 0:
            row = target_data.iloc[i]
            age = row["age"]
            distribution[age] += 1

    # Creating labels and values from the dictionary
    labels = list(distribution.keys())
    occurrences = list(distribution.values())

    # Creating the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, occurrences, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Occurrences')
    plt.title('Data Distribution')
    plt.savefig(f"{split}_data_distribution")


if __name__ == "__main__":
    main("train")