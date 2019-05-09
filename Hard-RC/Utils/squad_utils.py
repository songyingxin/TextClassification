import os

import torch
from torchtext import data
from torchtext import datasets
from torchtext import vocab

import nltk

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def load_squad(path, raw_field, word_field, label_field, train_batch_size, dev_batch_size, device, word_embedding_file, cache_dir):

    if os.path.exists(cache_dir):
        print("dataset have cached, loding splits... ")

        list_fields = [('id', raw_field), ('s_idx', label_field), ('e_idx', label_field),
                       ('context', word_field),
                       ('question', word_field)]
        train_examples = torch.load(cache_dir + 'train_examples.pt')
        dev_examples = torch.load(cache_dir + "dev_examples.pt")

        train = data.Dataset(examples=train_examples, fields=list_fields)
        dev = data.Dataset(examples=dev_examples, fields=list_fields)
    
    else:

        dict_field = {'id': ('id', raw_field),
                    's_idx': ('s_idx', label_field),
                    'e_idx': ('e_idx', label_field),
                    'context': ('context', word_field),
                    'question': ('question', word_field)}

        train, dev = data.TabularDataset.splits(
            path=path,
            train='train.jsonl',
            validation='dev.jsonl',
            format='json',
            fields=dict_field
        )

        os.makedirs(cache_dir)
        torch.save(train.examples, cache_dir + 'train_examples.pt')
        torch.save(dev.examples, cache_dir + "dev_examples.pt")

    print("the size of train: {}, dev:{}".format(
        len(train.examples), len(dev.examples)))

    word_field.build_vocab(train, dev, vectors=vocab.Vectors(
        word_embedding_file), max_size=25000, unk_init=torch.Tensor.normal_)

    print("building iterators...")

    train_iter, dev_iter = data.BucketIterator.splits(
        (train, dev), batch_sizes=[
            train_batch_size, dev_batch_size], device=device, sort_key=lambda x: len(x.c_word)
    )

    return train_iter, dev_iter



