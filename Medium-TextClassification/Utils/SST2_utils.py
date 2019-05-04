import torch
from torchtext import data
from torchtext import datasets
from torchtext import vocab


def load_sst2(path, text_field, label_field, batch_size, device, embedding_file):

    train, dev, test = data.TabularDataset.splits(
        path=path, train='train.tsv', validation='dev.tsv',
        test='test.tsv', format='tsv', skip_header=True,
        fields=[('text', text_field), ('label', label_field)])
    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(dev.examples), len(test.examples)))
    vectors = vocab.Vectors(embedding_file)

    text_field.build_vocab(
        train, dev, test, max_size=25000,
        vectors=vectors, unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, shuffle=True, device=device
    )

    return train_iter, dev_iter, test_iter

