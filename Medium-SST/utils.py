import spacy
import time
import matplotlib.pyplot as plt
import csv

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

def sst_analysis(filename, type):
    text_lengths = []
    with open(filename, 'r', encoding='utf-8') as fh:
        rowes = csv.reader(fh, delimiter='\t')
        for row in rowes:
            text = row[0]
            text_lengths.append(len(word_tokenize(text)))
    
    x = range(len(text_lengths))
    plt.plot(x, text_lengths)
    plt.ylabel("tokens_num")
    plt.title(type)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    sst_analysis('data/SST2/train.tsv', 'train')
    sst_analysis('data/SST2/dev.tsv', 'dev')
    sst_analysis('data/SST2/test.tsv', 'test')
    
