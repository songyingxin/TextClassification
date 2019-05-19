import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM
from models.Embedding import Embedding

class LSTMATTHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(LSTMATTHighway, self).__init__()

        self.char_embedding = nn.Embedding.from_pretrained(
            char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(
            word_emb, freeze=False)

        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)

        self.rnn = LSTM(word_dim + char_dim, hidden_size,
                        num_layers, bidirectional, dropout)

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, text_word, text_char):
        text_word, text_lengths = text_word

        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))

        char_emb = char_emb.permute(1, 0, 2, 3)

        text_emb = self.text_embedding(word_emb, char_emb)

        hidden, outputs = self.rnn(text_emb, text_lengths)
        # hidden； [num_layers * bidirectional, batch_size, hidden_size]
        # outputs: [real_seq_len, batch_size, hidden_size * 2]

        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, real_seq, hidden_size * 2]

        """ tanh attention 的实现 """
        score = torch.tanh(torch.matmul(outputs, self.W_w))
        # score: [batch_size, real_seq, hidden_size * 2]

        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, real_seq, 1]

        scored_x = outputs * attention_weights
        # scored_x : [batch_size, real_seq, hidden_size * 2]

        feat = torch.sum(scored_x, dim=1)
        # feat : [batch_size, hidden_size * 2]

        return self.fc(feat)
