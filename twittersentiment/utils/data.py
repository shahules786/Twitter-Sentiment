import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):

    """

    Twitter dataset

    """

    def __init__(self, train, targets=None, mode="train"):

        self.train = train
        self.mode = mode
        self.targets = targets

    def __len__(self):

        return len(self.train)

    def __getitem__(self, idx):

        x_train_fold = torch.tensor(self.train[idx], dtype=torch.long).to(DEVICE)

        if self.mode == "train":
            y_train_fold = torch.tensor(self.targets[idx], dtype=torch.float32).to(DEVICE)
            return x_train_fold, y_train_fold
        else:
            return x_train_fold, 0
