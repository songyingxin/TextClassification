import random
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from torchtext import vocab

import args
import models

def imdb_data(config, text_field, label_field, device):
    if config.has_data:
        train_data, test_data = datasets.IMDB.splits(text_field, label_field, root=config.imdb)
    else:
        train_data, test_data = datasets.IMDB.splits(text_field, label_field)

    test_data, dev_data = test_data.split(random_state=random.seed(config.seed))
    
    if config.has_embedding:
        vectors = vocab.Vectors(config.glove_word_file)
    else:
        vectors = config.embedding_name

    text_field.build_vocab(
        train_data, max_size=config.glove_word_size, 
        vectors=vectors,unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_size = config.batch_size,
        sort_within_batch=True,
        device = device
    )

    return train_iterator, dev_iterator, test_iterator


def binary_accuracy(preds, y):
    """
    精确度计算
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, model_name="RNN"):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.text
        if model_name == "RNN":
            predictions = model(text, text_lengths).squeeze(1)
        elif model_name == "CNN":
            predictions = model(text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, model_name="RNN"):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text
            if model_name == "RNN":
                predictions = model(text, text_lengths).squeeze(1)
            elif model_name == "CNN":
                predictions = model(text).squeeze(1)


            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main(config):

    """ 设备准备： cpu 或 gpu  """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True # cudnn 使用确定性算法，保证每次结果一样

    """ 数据准备：imdb, 采用torchtext """
    text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True)
    label_field = data.LabelField(dtype=torch.float)

    train_iterator, dev_iterator, test_iterator = imdb_data(config, text_field, label_field, device)

    """ 模型，词向量准备 """
    pretrained_embeddings = text_field.vocab.vectors
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]

    if config.model_name == "RNN":
        model = models.RNN(
            len(text_field.vocab), config.glove_word_dim, config.output_dim, 
            config.hidden_size, config.num_layers, config.bidirectional, config.dropout, pad_idx=pad_idx)
    elif config.model_name == "CNN":
        filter_sizes = config.filter_sizes
        filter_sizes = [int(val) for val in filter_sizes.split()]
        model = models.CNN(
            len(text_field.vocab), config.glove_word_dim, config.filter_num, filter_sizes,
            config.output_dim, config.dropout, pad_idx)
    
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.rand(config.glove_word_dim)
    model.embedding.weight.data[pad_idx] = torch.rand(config.glove_word_dim)

    """ 开始训练 """
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    for epoch in range(config.epoch_num):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, config.model_name)
        valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, config.model_name)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion, config.model_name)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


if __name__ == "__main__":
    main(args.get_args())
