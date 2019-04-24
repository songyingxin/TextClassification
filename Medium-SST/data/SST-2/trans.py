import csv
import json


def trans(input_file, output_file):

    dump = []
    total = 1
    with open(input_file, 'r', encoding='utf-8') as fh:
        rowes = csv.reader(fh, delimiter='\t')
        for row in rowes:
            idx = str(total)
            label = int(row[1])
            text = row[0]

            dump.append(dict([
                ('idx', idx),
                ('text', text),
                ('label', label)
            ]))
            total += 1
    
    with open(f'{output_file}l', 'w', encoding='utf-8') as f:
        for line in dump:
            json.dump(line, f)
            print('', file=f)


if __name__ == "__main__":
    trans("train.tsv", "train.json")
    trans("dev.tsv", "dev.json")
    trans("test.tsv", "test.json")
