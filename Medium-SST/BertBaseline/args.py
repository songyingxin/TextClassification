import argparse


def get_args():
    data_dir = "data/SST-2"
    output_dir = "data"

    parser = argparse.ArgumentParser(description='BERT Baseline')

    parser.add_argument(
        "--data_dir",default=data_dir,type=str,required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument(
        "--bert_model", default=bert-base-uncased, type=str, required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--output_dir",default=None,type=str,required=True,
        help="The output directory where the model predictions and checkpoints will be written.")


    config = parser.parse_args()

    return config
