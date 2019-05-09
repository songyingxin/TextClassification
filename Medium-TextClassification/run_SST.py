
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

from Utils.utils import word_tokenize, get_device, epoch_time, classifiction_metric
from Utils.SST2_utils import load_sst2


def train(model, iterator, optimizer, criterion, num_labels):

    epoch_loss = 0
    
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    
    model.train()
    for batch in iterator:

        optimizer.zero_grad()

        logits = model(batch.text)

        loss = criterion(logits.view(-1, num_labels), batch.label)
        
        labels = batch.label.detach().cpu().numpy()
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

        all_preds = np.append(all_preds, preds)
        all_labels = np.append(all_labels, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    report, acc = classifiction_metric(
        all_preds, all_labels,  ["negative", "positive"])
    return epoch_loss/len(iterator), acc, report


def evaluate(model, iterator, criterion, num_labels):

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    
    model.eval()

    with torch.no_grad():

        for batch in iterator:
            logits = model(batch.text)

            loss = criterion(logits.view(-1, num_labels), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    report, acc = classifiction_metric(all_preds, all_labels, ["negative", "positive"])

    return epoch_loss/len(iterator), acc, report




def main(config):
    print("\t \t \t the model name is {}".format(config.model_name))
    device, n_gpu = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样

    """ sst2 数据准备 """
    text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True, fix_length=config.sequence_length)
    label_field = data.LabelField(dtype=torch.long)

    train_iterator, dev_iterator, test_iterator = load_sst2(config.data_path, text_field, label_field, config.batch_size, device, config.glove_word_file)

    """ 词向量准备 """
    pretrained_embeddings = text_field.vocab.vectors


    """ 模型准备 """
    if config.model_name == "TextCNN":
        filter_sizes = [int(val) for val in config.filter_sizes.split()]
        model = TextCNN.TextCNN(config.glove_word_dim, config.filter_num, filter_sizes,
            config.output_dim, config.dropout, pretrained_embeddings)
    elif config.model_name == "TextRNN":
        model = TextRNN.TextRNN(config.glove_word_dim, config.output_dim,
                            config.hidden_size, config.num_layers, config.bidirectional, config.dropout, pretrained_embeddings)
    elif config.model_name == "LSTMATT":
        model = LSTMATT.LSTMATT(config.glove_word_dim, config.output_dim,
                            config.hidden_size, config.num_layers, config.bidirectional, config.dropout, pretrained_embeddings)
    elif config.model_name == 'TextRCNN':
        model = TextRCNN.TextRCNN(config.glove_word_dim, config.output_dim,
                                  config.hidden_size, config.num_layers, config.bidirectional, config.dropout, pretrained_embeddings)


    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_dev_loss = float('inf')
    for epoch in range(config.epoch_num):
        start_time = time.time()

        train_loss, train_acc, train_report = train(
            model, train_iterator, optimizer, criterion, config.output_dim)
        dev_loss, dev_acc, dev_report = evaluate(
            model, dev_iterator, criterion, config.output_dim)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'---------------- Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s ----------')
        print("-------------- Train -------------")
        print(f'\t \t Loss: {train_loss:.3f} |  Acc: {train_acc*100: .2f} %')
        print('{}'.format(train_report))
        print("-------------- Dev -------------")
        print(f'\t \t Loss: {dev_loss: .3f} | Acc: {dev_acc*100: .2f} %')
        print('{}'.format(dev_report))

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc, test_report = evaluate(
        model, test_iterator, criterion, config.output_dim)
    print("-------------- Test -------------")
    print(f'\t \t Loss: {test_loss: .3f} | Acc: {test_acc*100: .2f} %')
    print('{}'.format(test_report))


if __name__ == "__main__":

    model_name = "TextCNN"   # TextRNN, TextCNN， lSTMATT, TextRCNN
    data_dir = "/home/songyingxin/datasets/SST-2"
    cache_dir = data_dir + "/cache/"
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/"

    if model_name == "TextCNN":
        from TextCNN import args, TextCNN
        main(args.get_args(data_dir, cache_dir, embedding_folder))
    elif model_name == "TextRNN":
        from TextRNN import args, TextRNN
        main(args.get_args(data_dir, cache_dir, embedding_folder))
    elif model_name == "LSTMATT":
        from LSTM_ATT import args, LSTMATT
        main(args.get_args(data_dir, cache_dir, embedding_folder))
    elif model_name == "TextRCNN":
        from TextRCNN import args, TextRCNN
        main(args.get_args(data_dir, cache_dir, embedding_folder))


    
    
