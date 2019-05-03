import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        """
        :param query: [dim, 1]
        :param key: [batch_size, key_len, dim]
        :param value: [batch_size, key_len, dim]
        """
        score = torch.tanh(torch.matmul(key, ))


        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        return torch.matmul(p_attn, value), p_attn




        


