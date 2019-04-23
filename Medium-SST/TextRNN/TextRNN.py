import torch.nn as nn
import torch.nn.functional as F
import torch

class TextRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pad_idx):
        super(TextRNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(
                    getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(
                    getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(
                    getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(
                    getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        text, text_lengths = x 
        # text: [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [sent len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        # [层数 * 单向/双向, batch_size, hidden_size]
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        return self.fc(hidden.squeeze(0))


