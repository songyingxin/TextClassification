import torch
import torch.nn as nn
import torch.nn.functional as F

from .LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    residual connection + layer norma lization
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        """
        return x + self.dropout(sublayer(self.norm(x)))

