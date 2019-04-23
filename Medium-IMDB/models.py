import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pad_idx):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, 
            bidirectional=bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, text, text_lengths):

        # text: [sent len, batch size]
        embedded = self.dropout(self.embedding(text))  
        # embedded: [sent len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded) # [层数 * 单向/双向, batch_size, hidden_size]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        return self.fc(hidden.squeeze(0))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text: [sent len, batch size]
        text = text.permute(1, 0)  # 维度换位, 
        # text: [batch size, sent len]

        embedded = self.embedding(text)  
        # embedded: [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # 在第1维增加一个维度值为 1 的维度
        #embedded：[batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conv_n： [batch size, n_filters, sent len - filter_sizes[n]]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)
