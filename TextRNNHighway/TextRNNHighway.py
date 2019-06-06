import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM
from models.Linear import Linear
from models.Embedding import Embedding

class TextRNNHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(TextRNNHighway, self).__init__()

        self.char_embedding = nn.Embedding.from_pretrained(
            char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(
            word_emb, freeze=False)

        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)

        self.rnn = LSTM(word_dim + char_dim, hidden_size,
                        num_layers, bidirectional, dropout)

        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, text_lengths = text_word

        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))
        
        char_emb = char_emb.permute(1, 0, 2, 3)

        text_emb = self.text_embedding(word_emb, char_emb)

        # embedded: [sent len, batch size, emb dim]
        hidden, outputs = self.rnn(text_emb, text_lengths)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        return self.fc(hidden)
