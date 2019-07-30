import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from Transformer.Embeddings import Embeddings
from Transformer.PositionalEncoding import PositionalEncoding
from Transformer.MultiHeadAttention import MultiHeadAttention
from Transformer.PositionwiseFeedForward import PositionwiseFeedForward
from Transformer.Encoder import Encoder
from Transformer.EncoderLayer import EncoderLayer

class TransformerText(nn.Module):
    """ 用 Transformer 来作为特征抽取的基本单元 """
    def __init__(self, head, n_layer, emd_dim, d_model, d_ff, output_dim, dropout, pretrained_embeddings):
        super(TransformerText, self).__init__()

        # # nn.Embedding()
        # self.word_embedding = Embeddings(d_model, 250000)
        # self.position_embedding = PositionalEncoding(d_model, dropout)

        # nn.Embedding.from_pretrain()
        self.word_embedding = Embeddings(pretrained_embeddings, emd_dim)
        self.position_embedding = PositionalEncoding(emd_dim, dropout)

        self.trans_linear = nn.Linear(emd_dim, d_model)

        multi_attn = MultiHeadAttention(head, d_model)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, multi_attn, feed_forward, dropout), n_layer)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        """
        x:
            text: [sent len, batch size], 文本数据
            text_lens: [batch_size], 文本数据长度
        """
        text, _ = x
        # text: [sent len, batch size]
        text = text.permute(1, 0)
        # text: [batch_size, sent_len]

        embeddings = self.word_embedding(text)
        # embeddings: [batch_size, sent_len, emd_dim]
        embeddings = self.position_embedding(embeddings)
        # embeddings: [batch_size, sent_len, dim]

        embeddings = self.trans_linear(embeddings)

        embeddings = self.encoder(embeddings)
        # embeddings: [batch_size, sent_len, d_model]
        
        features = embeddings[:, -1, :]
        # features: [batch_size, d_model]
        
        return self.fc(features)



        
