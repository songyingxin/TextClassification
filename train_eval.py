import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import time

from Utils.utils import classifiction_metric

def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir, print_step, data_type='word'):

    model.train()
    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime('%H:%M:%S', time.gmtime()))

    global_step = 0
    best_dev_loss = float('inf')

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch+1:02} ----------')

        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            optimizer.zero_grad()

            if data_type == 'word':
                logits = model(batch.text)
            elif data_type == 'highway':
                logits = model(batch.text_word, batch.text_char)

            loss = criterion(logits.view(-1, len(label_list)), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:

                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)
                
                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list, data_type)
                c = global_step // print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/dev", dev_loss, c)

                writer.add_scalar("acc/train", train_acc, c)
                writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                print_list = ['macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)
                
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), out_model_file)
                
                model.train()

    writer.close()


def evaluate(model, iterator, criterion, label_list, data_type='word'):
    model.eval()

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    

    with torch.no_grad():

        for batch in iterator:

            if data_type == 'word':
                with torch.no_grad():
                    logits = model(batch.text)
            elif data_type == 'highway':
                with torch.no_grad():
                    logits = model(batch.text_word, batch.text_char)

            loss = criterion(logits.view(-1, len(label_list)), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    acc, report = classifiction_metric(
        all_preds, all_labels, label_list)

    return epoch_loss/len(iterator), acc, report
