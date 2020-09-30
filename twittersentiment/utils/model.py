from torch import nn
import torch
import torch.nn.functional as F


class TwitterModel(nn.Module):
    def __init__(self, embedding_matrix, lstm_hidden_size=128, gru_hidden_size=64):

        super(TwitterModel, self).__init__()
        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.Linear1 = nn.Linear(gru_hidden_size * 4, 1)

    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def flatten_parameters(self):
        self.lstm.flatten_parameters()
        self.lstm2.flatten_parameters()

    def forward(self, x):

        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_lstm, _ = self.lstm2(h_lstm)

        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        concat = torch.cat((avg_pool, max_pool), 1)

        out = torch.sigmoid(self.Linear1(concat))
        return out
