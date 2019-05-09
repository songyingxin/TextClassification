import random
import time
import numpy as np
import argparse

import torch
import torch.optim as optim
import torch.nn as nn


from torchtext import data
from torchtext import datasets
from torchtext import vocab


from Utils.utils import get_device, word_tokenize, epoch_time, classifiction_metric
from Utils.squad_utils import load_squad

def main(config):

    print("\t \t \t the model name is {}".format(config.model_name))

    device, n_gpu = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样
    
    raw_field = data.RawField()
    raw_field.is_target = False
    word_field = data.Field(
        batch_first=True, tokenize=word_tokenize,
        lower=True, include_lengths=True)
    label_field = data.Field(sequential=False, unk_token=None, use_vocab=False)

    train_iterator, dev_iterator = load_squad(config.data_path, raw_field, word_field, label_field, config.train_batch_size, config.dev_batch_size, device, config.glove_word_file,  config.cache_path)

    pretrained_embeddings = word_field.vocab.vectors

    model = BIDAF.BIDAF(pretrained_embeddings, config.glove_wored_dim, config.hidden_size, config.num_layers, config.dropout)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)


    

    



if __name__ == "__main__":

    model_name = "BIDAF"

    data_dir = "/home/songyingxin/datasets/squad"
    cache_dir = ".cache"
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/"


    if model_name == "BIDAF":
        from BIDAF import args, BIDAF
        main(args.get_args(data_dir, cache_dir, embedding_folder))
