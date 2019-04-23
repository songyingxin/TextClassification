import argparse


def get_args():
    
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/"
    dataset_dir = '/home/songyingxin/datasets'

    parser = argparse.ArgumentParser(description='IMDB')

    parser.add_argument("--model_name", default="RNN", type=str, help="模型名称")

    #imdb 数据
    parser.add_argument("--has_data", default=True, type=bool, help="是否已经下载好数据集， 如果没有下载好，torchtext 会帮助你下载，新手推荐 False")
    parser.add_argument("--imdb", default=dataset_dir, type=str, help="imdb 数据集的地址")
    

    parser.add_argument("--seed", default=1234, type=int, help="随机种子")

    # 主要参数
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch_num", default=3, type=int)

    # word Embedding
    parser.add_argument("--has_embedding", default=True, type=bool, help="是否使用自己的词向量，如果不使用，torchtext会帮助你下载你指定的词向量")
    parser.add_argument("--embedding_name", default="glove.840B.300d")

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

    # rnn model
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.4, type=float)


    # cnn model
    parser.add_argument("--filter_num", default=3, type=int, help="filter 的数量")
    parser.add_argument("--filter_sizes", default="3 4 5", type=str, help="filter 的 size")


    config = parser.parse_args()

    return config
