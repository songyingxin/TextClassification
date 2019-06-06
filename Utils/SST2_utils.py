import torch
from torchtext import data
from torchtext import datasets
from torchtext import vocab


def load_sst2(path, text_field, label_field, batch_size, device, embedding_file, cache_dir):

    train, dev, test = data.TabularDataset.splits(
        path=path, train='train.tsv', validation='dev.tsv',
        test='test.tsv', format='tsv', skip_header=True,
        fields=[('text', text_field), ('label', label_field)])
    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(dev.examples), len(test.examples)))
    vectors = vocab.Vectors(embedding_file, cache_dir)

    text_field.build_vocab(
        train, dev, test, max_size=25000,
        vectors=vectors, unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, shuffle=True, device=device
    )

    return train_iter, dev_iter, test_iter


def sst_word_char(path, word_field, char_field, label_field, batch_size, device, word_emb_file, char_emb_file, cache_dir):

    fields = {
        'text': [('text_word', word_field), ('text_char', char_field)],
        'label': ('label', label_field)
    }
    train, dev, test = data.TabularDataset.splits(
        path=path, train='train.jsonl', validation='dev.jsonl',
        test='test.jsonl', format='json', skip_header=True,
        fields=fields)
    
    word_vectors = vocab.Vectors(word_emb_file, cache_dir)
    char_vectors = vocab.Vectors(char_emb_file, cache_dir)

    word_field.build_vocab(
        train, dev, test, max_size=25000,
        vectors=word_vectors, unk_init=torch.Tensor.normal_)
    char_field.build_vocab(
        train, dev, test, max_size=94,
        vectors=char_vectors, unk_init=torch.Tensor.normal_)
    
    label_field.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text_word), sort_within_batch=True, repeat=False, shuffle=True, device=device
    )

    return train_iter, dev_iter, test_iter






if __name__ == "__main__":

    data_dir = "/home/songyingxin/datasets/SST-2"

    CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
    char_field = data.NestedField(CHAR_NESTING, tokenize='spacy')
    word_field = data.Field(tokenize='spacy', lower=True,
                            include_lengths=True, fix_length=100)
    label_field = data.LabelField(dtype=torch.long)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    word_emb_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d.txt"
    char_emb_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d-char.txt"

    train_iter, dev_iter, test_iter = sst_word_char(
        data_dir, word_field, char_field, label_field, 32, device, word_emb_file, char_emb_file)
    
    for batch in train_iter:

        print(batch)






    
