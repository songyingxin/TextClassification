
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn


from torchtext import data
from torchtext import datasets
from torchtext import vocab

from TextCNN import args, TextCNN
from utils import word_tokenize, get_device, epoch_time 
from datasets import load_sst2


def binary_accuracy(preds, y):
    """
    精确度计算
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def main(config):

    device = get_device()

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样

    """ sst2 数据准备 """
    text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True, fix_length=40)
    label_field = data.LabelField(dtype=torch.float)

    train_iterator, dev_iterator, test_iterator = load_sst2(config.data_path, text_field, label_field, config.batch_size, device, config.glove_word_file)

    """ 词向量准备 """
    pretrained_embeddings = text_field.vocab.vectors
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]

    """ 模型准备 """
    filter_sizes = [int(val) for val in config.filter_sizes.split()]
    model = TextCNN.TextCNN(
        len(text_field.vocab), config.glove_word_dim, config.filter_num, filter_sizes,
        config.output_dim, config.dropout, pad_idx)
    
    # 模型填充词向量
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.rand(config.glove_word_dim)
    model.embedding.weight.data[pad_idx] = torch.rand(config.glove_word_dim)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    for epoch in range(config.epoch_num):
        start_time = time.time()

        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(
            model, dev_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc = evaluate(
        model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')












if __name__ == "__main__":
    main(args.get_args())
    
