import torch
import torch.nn as nn
import torch.nn.functional as F


class BIDAF(nn.Module):

    def __init__(self, args, pretrained):
        super(BIDAF, self).__init__()

        self.char_emb = nn.Embedding(args.glove_char_size, config.glove_char_dim, padding_idx=1)
        