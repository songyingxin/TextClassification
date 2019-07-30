import torch
import torch.nn as nn
import torch.nn.functional as F

from .ScaleDotProductAttention import ScaledDotProduction
from .utils import clones


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = ScaledDotProduction(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, Q_len, d_model]
            K: [batch_size, K_len, d_model]
            V: [batch-size, V_len, d_model]
            mask: 
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        batch_size = Q.size(0)  # batch-size

        Q, K, V = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for l,x in zip(self.linears, (Q, K, V))
        ]
        # Q: [batch_size, head, Q_len, d_model/head]
        # K: [batch_size, head, K_len, d_model/head]
        # V: [batch_size, head, V_len, d_model/head]

        x, attn = self.attn(Q, K, V, mask=mask)
        # x: [batch_size, head, Q_len, d_model/head]
        # attn: [batch_size, head, Q_len, K_len]

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # x: [batch_size, Q_len, d_model]
        return self.linears[-1](x)


