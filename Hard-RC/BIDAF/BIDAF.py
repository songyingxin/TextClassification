import torch
import torch.nn as nn
import torch.nn.functional as F


class BIDAF(nn.Module):

    def __init__(self, char_pretrained, word_pretrained):
        super(BIDAF, self).__init__()

        self.char_emb = nn.Embedding.from_pretrained(char_pretrained, freeze=False)
        