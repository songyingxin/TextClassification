import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clones
from .LayerNorm import LayerNorm

class Encoder(nn.Module):
    """ Transformer Encoder
    It includes N layer of EncoderLayer
    """
    def __init__(self, layer, N):
        """
        layer: EncoderLayer, 每层的网络
        N: Encoder 包含 N 层 EncoderLayer
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
