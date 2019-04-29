import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Conv import Conv
from models.Linear import Linear


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = Conv(embedding_dim,n_filters,filter_sizes)

        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        text, _ = x
        # text: [sent len, batch size]
        text = text.permute(1, 0)  # 维度换位,
        # text: [batch size, sent len]

        embedded = self.embedding(text)
        # embedded: [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, emb dim, sent len]

        conved = self.convs(embedded)

        #conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)
