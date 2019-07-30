import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clones
from .SublayerConnection import SublayerConnection

class EncoderLayer(nn.Module):
    """
    Encoder Layer:
        self-attention + feed-forward layer
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """

        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

