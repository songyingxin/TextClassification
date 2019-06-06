import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM
from models.Linear import Linear


class TextRCNN(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(TextRCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + embedding_dim, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        text, text_lengths = x
        # text: [seq_len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [seq_len, batch size, emb dim]

        outputs, _ = self.rnn(embedded)
        # outputs: [seq_len， batch_size, hidden_size * bidirectional]

        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * bidirectional]

        embedded = embedded.permute(1, 0, 2)
        # embeded: [batch_size, seq_len, embeding_dim]

        x = torch.cat((outputs, embedded), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]

        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        # y2: [batch_size, hidden_size * bidirectional, seq_len]

        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        # y3: [batch_size, hidden_size * bidirectional]

        return self.fc(y3)
