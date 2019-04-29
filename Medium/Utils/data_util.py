import spacy
import csv
import random
import argparse
from collections import Counter
from tqdm import tqdm
import numpy as np

import args

NLP = spacy.blank("en")


def word_tokenize(sent):
    """ 分词 """
    doc = NLP(sent)
    return [token.text for token in doc]

def sst_data(filename, data_type="train", word_counter=None, char_counter=None):
    print("Generating {} examples ...".format(data_type))
    examples = []
    total = 1
    with open(filename, 'r') as fh:
        rowes = csv.reader(fh, delimiter='\t')
        for row in tqdm(rowes):
            label = row[0]
            text = row[1]

            text_tokens = word_tokenize(text)
            text_chars = [list(token) for token in text_tokens]

            for token in text_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            example = {
                "text_tokens" : text_tokens,
                "text_chars": text_chars,
                "label" : int(label),
                "id": total
            }
            total += 1

            examples.append(example)
    print("the num of {} examples is {}".format(data_type, total))
    random.shuffle(examples)

    return examples


def get_embedding(counter, data_type,
                  emb_file=None, size=None, vec_size=None,
                  limit=-1, specials=["<PAD>", "<OOV>", "<SOS>", "<EOS>"]):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [
                np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    token2idx_dict = {token: idx
                      for idx, token
                      in enumerate(embedding_dict.keys(), len(specials))}
    for i in range(len(specials)):
        token2idx_dict[specials[i]] = i
        embedding_dict[specials[i]] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def filter_func(config, example):
    return len(example["text_tokens"]) > config.text_limit


def word2wid(word, word2idx_dict):
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
            return word2idx_dict[each]
    return word2idx_dict["<OOV>"]


def char2cid(char, char2idx_dict):
    if char in char2idx_dict:
        return char2idx_dict[char]
    return char2idx_dict["<OOV>"]

def build_features(config, examples, data_type,
                   word2idx_dict, char2idx_dict, debug=False):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    examples_with_features = []
    for example in tqdm(examples):
        total_ += 1
        if filter_func(config, example):
            continue
        total += 1

        text_wids = np.ones(
            [config.text_limit], dtype=np.int32) * \
            word2idx_dict["<PAD>"]
        text_cids = np.ones(
            [config.text_limit, config.char_limit], dtype=np.int32) * \
            char2idx_dict["<PAD>"]
        

        for i, token in enumerate(example["text_tokens"]):
            text_wids[i] = word2wid(token, word2idx_dict)

        for i, token in enumerate(example["text_chars"]):
            for j, char in enumerate(token):
                if j == config.char_limit:
                    break
                text_cids[i, j] = char2cid(char, char2idx_dict)

        item = {}

        item["text_wids"] = text_wids
        item["text_cids"] = text_cids
        item['label'] = example['label']
        item['id'] = example['id']

        examples_with_features.append(item)

    print("Built {} / {} instances of features in total".format(total, total_))
    return examples_with_features


def save(filepath, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def prepro(config):
    if config.dataset == "sst1":
        dir_name = config.sst1
    else:
        dir_name = config.sst2

    word_counter, char_counter = Counter(), Counter()

    train_examples = sst_data(
        dir_name + "train.tsv", data_type="train", word_counter=word_counter, char_counter=char_counter)
    dev_examples = sst_data(
        dir_name + "dev.tsv", data_type="dev",
        word_counter=word_counter, char_counter=char_counter)
    test_examples = sst_data(
        dir_name + "test.tsv", data_type="test",word_counter=word_counter, char_counter=char_counter)
    
    word_emb_file = config.glove_word_file
    word_emb_size = config.glove_word_size
    word_emb_dim = config.glove_word_dim

    char_emb_file = config.glove_char_file
    char_emb_size = config.glove_char_size
    char_emb_dim = config.glove_char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file,
        size=word_emb_size, vec_size=word_emb_dim)  # 词向量矩阵
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file,
        size=char_emb_size, vec_size=char_emb_dim)
    
    train_examples= build_features(
        config, train_examples, "train",
        word2idx_dict, char2idx_dict)
    dev_examples = build_features(
        config, dev_examples, "dev",
        word2idx_dict, char2idx_dict)
    test_examples = build_features(
        config, test_examples, "test",
        word2idx_dict, char2idx_dict)

if __name__ == "__main__":

    config = args.get_args()
    prepro(config)






