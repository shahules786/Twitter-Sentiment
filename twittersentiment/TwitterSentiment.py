import os
import torch
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from torch.utils import model_zoo
from twittersentiment.utils.model import TwitterModel as Model
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from torch.utils.data import DataLoader
from twittersentiment.utils.model import TwitterModel
from twittersentiment.utils.preprocessing import Preprocess
from twittersentiment.utils.data import TweetDataset


SEED = 2020
MAX_LEN = 80
torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sentiment:
    def __init__(self):

        self.device = DEVICE

        model = namedtuple("model", ["url", "model"])

        self.models = {
            "twitter-en": model(
                url="https://github.com/shahules786/ML_EXP/releases/download/sample/best_model.pt",
                model=Model,
            )
        }

        self.tokenizer_path = "/home/shahul/Downloads/tokenizer.pickle"

    def load_pretrained(self, model_name="twitter-en"):

        # state_dict = torch.hub.load_state_dict_from_url(self.models[model_name].url, progress=True, map_location="cpu")
        state_dict = torch.load("/home/shahul/Downloads/classifier.pt", map_location=torch.device("cpu"))
        self.model = self.models[model_name].model(state_dict["embedding.weight"].numpy())
        self.model.load_state_dict(state_dict)

    def train(self, X, y, epochs: int, lr: float, batch_size: int, path: str):

        X = pd.Series(check_array(X, dtype=str, ensure_2d=False))

        y = pd.Series(check_array(y, dtype=str, ensure_2d=False))

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have same number of samples")

        ## check if y contains [0,1]

        corpus = Preprocess.preprocess_text(X)
        padded_tweets, word_index, self.tokenizer_path = Preprocess.tokenizer(corpus)
        embedding_matrix = Preprocess.prepare_matrix(word_index)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        train_data = TweetDataset(X_train, y_train)
        test_data = TweetDataset(X_test, y_test)

        dataloaders = {
            "train": DataLoader(train_data, batch_size=batch_size, shuffle=True),
            "valid": DataLoader(test_data, batch_size=batch_size, shuffle=False),
        }

        model = TwitterModel(embedding_matrix).to(DEVICE)
        loss_fn = torch.nn.BCELoss().cuda()
        optimizer = torch.optim.Adam(model.parameters())

        best_loss = {"train": np.inf, "valid": np.inf}

        for epoch in range(epochs):

            epoch_loss = {"train": 0.00, "valid": 0.00}

            for phase in ["train", "valid"]:

                if phase == "train":
                    model = model.train()
                else:
                    model = model.eval()

                running_loss = 0.00

                for i, (x, y) in enumerate(dataloaders[phase]):

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):

                        predict = model(x).squeeze()
                        loss = loss_fn(predict, y)

                        if phase == "train":

                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() / len(dataloaders[phase])

                    epoch_loss[phase] = running_loss

            logging.info(
                "Epoch {}/{}   -   loss: {:5.5f}   -   val_loss: {:5.5f}".format(
                    epoch + 1, epochs, epoch_loss["train"], epoch_loss["valid"]
                )
            )

            if epoch_loss["valid"] < best_loss["valid"]:

                logging.info("saving model...")
                best_loss = epoch_loss

            torch.save(model.state_dict(), os.path.join(path, "best_model.pt"))

    def predict(self, X):

        if isinstance(X, str):
            X = np.array([X])
        elif isinstance(X, pd.Series):
            X = X.values
        else:
            X = check_array(X, dtype=str, ensure_2d=False)

        corpus = Preprocess.preprocess_text(X)
        padded_tweets, _w, _t = Preprocess.tokenizer(corpus, self.tokenizer_path, "test")
        predictions = []
        for tweet in padded_tweets:

            with torch.no_grad():

                prediction = self.model(tweet)

            predictions.append(prediction)

        return predictions
