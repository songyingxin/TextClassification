
import json
import nltk


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def preprocess_file(input_file, output_file):

    dump = []
    abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data['data']

        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                tokens = word_tokenize(context)
                for qa in paragraph['qas']:
                    id = qa['id']
                    question = qa['question']
                    for ans in qa['answers']:
                        answer = ans['text']
                        s_idx = ans['answer_start']
                        e_idx = s_idx + len(answer)

                        l = 0
                        s_found = False
                        for i, t in enumerate(tokens):
                            while l < len(context):
                                if context[l] in abnormals:
                                    l += 1
                                else:
                                    break
                            # exceptional cases
                            if t[0] == '"' and context[l:l + 2] == '\'\'':
                                t = '\'\'' + t[1:]
                            elif t == '"' and context[l:l + 2] == '\'\'':
                                t = '\'\''

                            l += len(t)
                            if l > s_idx and s_found == False:
                                s_idx = i
                                s_found = True
                            if l >= e_idx:
                                e_idx = i
                                break

                        dump.append(dict([('id', id),
                                            ('context', context),
                                            ('question', question),
                                            ('answer', answer),
                                            ('s_idx', s_idx),
                                            ('e_idx', e_idx)]))

        with open(f'{output_file}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)

    

if __name__ == "__main__":
    
    data_dir = "/home/songyingxin/datasets/squad"

    train_name = 'train-v1.1.json'
    dev_name = "dev-v1.1.json"

    out_train_name = "train.json"
    out_dev_name = "dev.json"

    preprocess_file(f'{data_dir}/{train_name}', f'{data_dir}/{out_train_name}')
    preprocess_file(f'{data_dir}/{dev_name}', f'{data_dir}/{out_dev_name}')
