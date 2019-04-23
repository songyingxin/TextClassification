import spacy
import time

import torch

from torchtext import data
from torchtext import datasets
from torchtext import vocab

NLP = spacy.blank("en")


def word_tokenize(sent):
    """ 分词 """
    doc = NLP(sent)
    return [token.text for token in doc]

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



