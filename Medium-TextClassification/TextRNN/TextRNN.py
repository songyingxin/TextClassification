import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM


class TextRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pad_idx):
        super(TextRNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = LSTM(embedding_dim, hidden_size, num_layers,bidirectional, dropout)

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        text, text_lengths = x
        # text: [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [sent len, batch size, emb dim]

        hidden = self.rnn(embedded, text_lengths)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出
            
        return self.fc(hidden)
