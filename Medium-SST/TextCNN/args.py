import argparse


def get_args():
    data_dir = "data/SST2"
    cache_dir = data_dir +"/cache/"
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/"

    parser = argparse.ArgumentParser(description='SST')

    parser.add_argument("--seed", default=1234, type=int, help="随机种子")

    # data_util
    parser.add_argument(
        "--data_path", default=data_dir, type=str, help="sst2 数据集位置")
    parser.add_argument(
        "--cache_path", default=cache_dir, type=str, help="数据缓存地址"
    )

    # 优化参数
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch_num", default=4, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)

    # 模型参数
    parser.add_argument("--output_dim", default=1, type=int)

    # TextCNN 参数
    parser.add_argument("--filter_num", default=100, type=int, help="filter 的数量")
    parser.add_argument("--filter_sizes", default="1 2 3 4 5",
                        type=str, help="filter 的 size")


    # word Embedding
    parser.add_argument(
        '--glove_word_file',
        default=embedding_folder + 'glove/glove.840B.300d.txt',
        type=str, help='path of word embedding file')
    parser.add_argument(
        '--glove_word_size',
        default=int(2.2e6), type=int,
        help='Corpus size for Glove')
    parser.add_argument(
        '--glove_word_dim',
        default=300, type=int,
        help='word embedding size (default: 300)')

    # char embedding
    parser.add_argument(
        '--glove_char_file',
        default=embedding_folder + "glove/glove.840B.300d-char.txt",
        type=str, help='path of char embedding file')
    parser.add_argument(
        '--glove_char_size',
        default=94, type=int,
        help='Corpus size for char embedding')
    parser.add_argument(
        '--glove_char_dim',
        default=300, type=int,
        help='char embedding size (default: 64)')
    

    config = parser.parse_args()

    return config
