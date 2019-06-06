import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM
from models.Linear import Linear
from models.Embedding import Embedding

class TextRCNNHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(TextRCNNHighway, self).__init__()

        self.char_embedding = nn.Embedding.from_pretrained(
            char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(
            word_emb, freeze=False)

        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)

        self.rnn = nn.LSTM(word_dim + char_dim, hidden_size, num_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + word_dim + char_dim, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):

        text_word, text_lengths = text_word

        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))

        char_emb = char_emb.permute(1, 0, 2, 3)

        text_emb = self.text_embedding(word_emb, char_emb)

        outputs, _ = self.rnn(text_emb)
        # outputs: [seq_len， batch_size, hidden_size * bidirectional]

        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * bidirectional]

        text_emb = text_emb.permute(1, 0, 2)
        # embeded: [batch_size, seq_len, embeding_dim]

        x = torch.cat((outputs, text_emb), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]

        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        # y2: [batch_size, hidden_size * bidirectional, seq_len]

        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        # y3: [batch_size, hidden_size * bidirectional]

        return self.fc(y3)
