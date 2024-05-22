import torch
import torch.utils
import torch.utils.data
import datasets
from utils import set_seed, seed_worker

def main(seed):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    dataset = datasets.__dict__["AgeDB"]
    train_dataset = dataset(
        seed=seed,
        data_folder="./datasets/AgeDB",
        aug="crop,flip,color,grayscale",
        split='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    for i in range(3):
        for idx, data_dict in enumerate(train_loader):
            if idx == 0:
                print(data_dict["label"])



if __name__ == "__main__":
    main(0)