import torch.nn as nn
import torch.nn.functional as F
import torch

from models.LSTM import LSTM

class LSTMATT(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(LSTMATT, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)
        self.rnn = LSTM(embedding_dim, hidden_size,
                        num_layers, bidirectional, dropout)

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)


    def forward(self, x):
        text, text_lengths = x
        # text: [seq_len, batch_size]
        # text_lengths : [batch_size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [seq_len, batch size, emb_dim]

        hidden, outputs = self.rnn(embedded, text_lengths)
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
