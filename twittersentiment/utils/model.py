import torch
from torch import nn
import torch.nn.functional as F


class TweetModel(nn.Module):
    def __init__(self, embedding_matrix, lstm_hidden_size=200, gru_hidden_size=128):

        super(TweetModel, self).__init__()
        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = True
        self.embedding_dropout = nn.Dropout2d(0.1)

        self.gru = nn.GRU(
            embedding_matrix.shape[1], gru_hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )

        self.dropout2 = nn.Dropout(0.25)
        self.Linear1 = nn.Linear(gru_hidden_size * 5, 16)
        self.Linear2 = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)

        x, (x_h, x_c) = self.gru(h_embedding)

        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        concat = torch.cat((avg_pool, x_h, max_pool), 1)
        concat = self.Linear1(concat)
        out = torch.sigmoid(self.Linear2(concat))
        return out
