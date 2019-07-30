import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ScaledDotProduction(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0.1):
        super(ScaledDotProduction, self).__init__()

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """ Q_len == K_len == V_len
        Args:
            Q: [batch_size, Q_len, dim]
            K: [batch_size, K_len, dim]
            V: [batch_size, V_len, dim]
            mask: 是否 mask， 只有 decoder 才需要
        Returns:
            output: [batch_size, Q_len, dim], attention value
            attn: [batch_size, Q_len, K_len]， scores 
        """
        d_k = Q.size(-1)  # dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: [batch_size, Q_len, K_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        # [batch_size, Q_len, K_len]

        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        # output: [batch_size, Q_len, dim]

        return output, attn


