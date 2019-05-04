import argparse


def get_args(data_dir, cache_dir, embedding_folder):

    parser = argparse.ArgumentParser(description='SST')

    parser.add_argument("--model_name", default="TextRCNN",
                        type=str, help="这批参数所属的模型的名字")
    parser.add_argument("--seed", default=1234, type=int, help="随机种子")

    # data_util
    parser.add_argument(
        "--data_path", default=data_dir, type=str, help="sst2 数据集位置")
    parser.add_argument(
        "--cache_path", default=cache_dir, type=str, help="数据缓存地址"
    )
    parser.add_argument(
        "--sequence_length", default=60, type=int, help="句子长度"
    )

    # 优化参数
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch_num", default=2, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)

    # 模型参数
    parser.add_argument("--output_dim", default=2, type=int)

    # TextRNN 参数
    parser.add_argument("--hidden_size", default=200, type=int, help="隐层特征维度")
    parser.add_argument('--num_layers', default=2, type=int, help='RNN层数')
    parser.add_argument("--bidirectional", default=True, type=bool)


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

    config = parser.parse_args()

    return config
