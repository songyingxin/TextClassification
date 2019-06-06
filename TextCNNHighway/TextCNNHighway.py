import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Conv import Conv1d
from models.Linear import Linear
from models.Embedding import Embedding

class TextCNNHighway(nn.Module):
    def __init__(self, word_dim, char_dim, n_filters, filter_sizes, output_dim,
                 dropout, word_emb, char_emb, highway_layers):

        super().__init__()

        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(
            word_emb, freeze=False)
        
        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)

        self.convs = Conv1d(word_dim + char_dim, n_filters, filter_sizes)

        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, _ = text_word

        word_emb = self.word_embedding(text_word)
        char_emb = self.char_embedding(text_char)

        char_emb = char_emb.permute(1, 0, 2, 3)

        text_emb = self.text_embedding(word_emb, char_emb)
        # text_emb: [sent len, batch size, emb dim]

        text_emb = text_emb.permute(1, 2, 0)
        # text_emb: [batch size, emb dim, sent len]

        conved = self.convs(text_emb)

        #conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)
