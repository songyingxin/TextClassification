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



# path = "/home/songyingxin/datasets/SST2"
# embedding_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d.txt"
# text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True)
# label_field = data.Field(sequential=False)

# load_sst(path, text_field, label_field, 32, embedding_file)

# path = "/home/songyingxin/datasets/SST2"

# RAW = data.RawField()
# CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
# CHAR = data.NestedField(CHAR_NESTING, tokenize=word_tokenize)
# WORD = data.Field(
#         batch_first=True, tokenize=word_tokenize,
#         lower=True, include_lengths=True)
# LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

# dict_fields = {
#         'idx': ('idx', RAW),
#         'text': [('word', WORD), ('char', CHAR)],
#         'label': ('label', LABEL)
# }
# train_file = "train.json"
# dev_file = "dev.json"
# test_file = "test.json"
# train, dev, test = data.TabularDataset.splits(path=path, train=f'{train_file}l', validation=f'{dev_file}l', test=f'{test_file}l', format='json', fields=dict_fields)
# print(len(train))

# char_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d-char.txt"
# word_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d.txt"
# char_vectors = vocab.Vectors(char_file)
# word_vectors = vocab.Vectors(word_file)

# CHAR.build_vocab(
#     train, dev, test, max_size=94,
#     vectors=char_vectors, unk_init=torch.Tensor.normal_)
# WORD.build_vocab(
#     train, dev, test, max_size=250000,
#     vectors=word_vectors, unk_init=torch.Tensor.normal_)
