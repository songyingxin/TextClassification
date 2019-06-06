
import random
import time
import numpy as np
import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn


from torchtext import data
from torchtext import datasets
from torchtext import vocab

from Utils.utils import word_tokenize, get_device, epoch_time, classifiction_metric
from Utils.SST2_utils import sst_word_char

from train_eval import train, evaluate


def main(config):

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print("\t \t \t the model name is {}".format(config.model_name))
    device, n_gpu = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样

    """ sst2 数据准备 """
    CHAR_NESTING = data.Field(tokenize=list, lower=True)
    char_field = data.NestedField(
        CHAR_NESTING, tokenize='spacy', fix_length=config.sequence_length)
    word_field = data.Field(tokenize='spacy', lower=True,
                            include_lengths=True, fix_length=config.sequence_length)
    label_field = data.LabelField(dtype=torch.long)


    train_iterator, dev_iterator, test_iterator =  sst_word_char(
        config.data_path, word_field, char_field, label_field, config.batch_size, device, config.glove_word_file, config.glove_char_file, config.cache_path)

    """ 词向量准备 """
    word_embeddings = word_field.vocab.vectors
    char_embeddings = char_field.vocab.vectors

    model_file = config.model_dir + 'model1.pt'

    """ 模型准备 """
    if config.model_name == "TextRNNHighway":
        from TextRNNHighway import TextRNNHighway
        model = TextRNNHighway.TextRNNHighway(
            config.glove_word_dim, config.glove_char_dim, config.output_dim,
            config.hidden_size, config.num_layers, config.bidirectional, 
            config.dropout, word_embeddings, char_embeddings, config.highway_layers)
    elif config.model_name == "TextCNNHighway":
        from TextCNNHighway import TextCNNHighway
        filter_sizes = [int(val) for val in config.filter_sizes.split()]
        model = TextCNNHighway.TextCNNHighway(config.glove_word_dim, config.glove_char_dim, config.filter_num, filter_sizes, config.output_dim, config.dropout, word_embeddings, char_embeddings, config.highway_layers)
    elif config.model_name == "LSTMATTHighway":
        from LSTMATTHighway import LSTMATTHighway
        model = LSTMATTHighway.LSTMATTHighway(config.glove_word_dim, config.glove_char_dim, config.output_dim, config.hidden_size, config.num_layers, config.bidirectional, config.dropout, word_embeddings, char_embeddings, config.highway_layers)
    elif config.model_name == "TextRCNNHighway":
        from TextRCNNHighway import TextRCNNHighway
        model = TextRCNNHighway.TextRCNNHighway(config.glove_word_dim, config.glove_char_dim, config.output_dim, config.hidden_size, config.num_layers, config.bidirectional, config.dropout, word_embeddings, char_embeddings, config.highway_layers)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)


    if config.do_train:

        train(config.epoch_num, model, train_iterator, dev_iterator, optimizer, criterion, ['0', '1'], model_file, config.log_dir, config.print_step, 'highway')

    model.load_state_dict(torch.load(model_file))
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_report = evaluate(
        model, test_iterator, criterion, ['0', '1'], 'highway')
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Micro avg F1: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['micro avg']['f1-score'], 
        test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))


if __name__ == "__main__":

    model_name = "TextRCNNHighway"   # TextRNN, TextCNN， lSTMATT, TextRCNN
    data_dir = "/home/songyingxin/datasets/SST-2"
    cache_dir = ".cache/"
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/glove/"

    model_dir = ".models/"
    log_dir = ".log/"

    if model_name == "TextCNNHighway":
        from TextCNNHighway import args, TextCNNHighway
        main(args.get_args(data_dir, cache_dir,
                           embedding_folder, model_dir, log_dir))
    elif model_name == "TextRNNHighway":
        from TextRNNHighway import args, TextRNNHighway
        main(args.get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir))
    elif model_name == "LSTMATTHighway":
        from LSTMATTHighway import args, LSTMATTHighway
        main(args.get_args(data_dir, cache_dir,
                           embedding_folder, model_dir, log_dir))
    elif model_name == "TextRCNNHighway":
        from TextRCNNHighway import args, TextRCNNHighway
        main(args.get_args(data_dir, cache_dir,
                           embedding_folder, model_dir, log_dir))
