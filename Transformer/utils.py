import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def clones(module, N):
    """
    clone N 个完全相同的 module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
