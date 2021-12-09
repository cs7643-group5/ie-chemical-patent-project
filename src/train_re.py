from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import pickle
import argparse
from tqdm import tqdm

from re_models import custom_model
from re_models import R_bert
from preprocessing import ee_preprocessor
# from preprocessing.ee_preprocessor import PatentDataset
# from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# count the number of trainable parameters in the model
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Universal function to compare predicted tags and actual tags
def measure_f1(model, dataloader, device, i2label):
    model.eval()

    results_dict = {v: {'tp': 0, 'fp': 0, 'fn': 0} for v in i2label.values()}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            sentences, masks, labels, trig_mask, ent_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
            output = model(sentences.to(device), masks.to(device), trig_mask, ent_mask)
            output = F.softmax(output, dim=1)
            output_class = torch.argmax(output, dim=1)

            for pred, actual in zip(output_class, labels):

                pred = i2label[pred.item()]
                actual = i2label[actual.item()]

                if pred == actual and pred != 'O':
                    results_dict[actual]['tp'] += 1
                elif not (pred == 'O' and actual == 'O'):
                    if pred == 'O':
                        results_dict[actual]['fn'] += 1
                    elif actual == 'O':
                        results_dict[pred]['fp'] += 1
                    else:
                        results_dict[actual]['fn'] += 1
                        results_dict[pred]['fp'] += 1
                else:
                    pass

    f1_scores = []

    tp_total = 0
    fn_total = 0
    fp_total = 0
    print('\n')
    for label in i2label.values():
        if label != 'O':
            precision = round(
                results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fp'] + 1e-6), 4)
            recall = round(results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fn'] + 1e-6),
                           4)
            f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)
            f1_scores.append(f1)
            print("Entity Label: ", label, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)
            tp_total += results_dict[label]['tp']
            fn_total += results_dict[label]['fn']
            fp_total += results_dict[label]['fp']

    precision_overall = round(tp_total / (tp_total + fp_total + 1e-6), 4)
    recall_overall = round(tp_total / (tp_total + fn_total + 1e-6), 4)
    f1_overall = round((2 * precision_overall * recall_overall) / (precision_overall + recall_overall + 1e-6), 4)
    f1_average = sum(f1_scores) / len(f1_scores)
    print('Average F1 Score: ', round(f1_average, 4))
    print("Overall Performance, Precision: ", precision_overall, ", Recall: ", recall_overall, ", F1: ", f1_overall)

    # for label in i2label.values():
    #     if label != 'O':
    #         precision = round(results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fp'] + 1e-6), 4)
    #         recall = round(results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fn'] + 1e-6), 4)
    #         f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)
    #         f1_scores.append(f1)
    #         print("Entity Label: ", label, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

    # f1_average = sum(f1_scores) / len(f1_scores)
    # print('Average F1 Score: ', round(f1_average, 4))
    return f1_average


# calculate the loss from the model on the provided dataloader
def evaluate(model,
             dataloader,
             device):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            sentences, masks, labels, trig_mask, ent_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
            output = model(sentences.to(device), masks.to(device), trig_mask, ent_mask)
            loss = F.cross_entropy(output, labels.to(device))

            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# calculate the prediction accuracy on the provided dataloader
def evaluate_acc(model,
                 dataloader,
                 device):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        total_correct = 0
        total = 0
        for i, batch in enumerate(dataloader):
            sentences, masks, labels, trig_mask, ent_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
            output = model(sentences.to(device), masks.to(device), trig_mask, ent_mask)
            output = F.softmax(output, dim=1)
            output_class = torch.argmax(output, dim=1)
            total_correct += torch.sum(torch.where(output_class == labels.to(device), 1, 0))
            total += sentences.size()[0]

    return total_correct / total


# computes the amount of time that a training epoch took and displays it in human readable form
def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_classification_head_weights(m: nn.Module, hidden_size=768):
    k = 1 / hidden_size
    # these are the names of the added layers in the added
    # classficiation head
    class_head_params = ['linear_1_class',
                         'relu_class',
                         'linear_2_class',
                         'dropout_layer_class']
    # loop through all model parameter names
    for name, param in m.named_parameters():
        # the names to look out for will be something like
        # linear_1_class.weight or linear_1_class.bias so
        # need to iterate over all the names in class_head_params and see
        # if any are contained within the model parameter name
        if any([class_name in name for class_name in class_head_params]):
            # apply respective intilization depending on if the name is weight
            # or bias (captured in else)
            if 'weight' in name:
                print(name)
                nn.init.uniform_(param.data, a=-1 * k ** 0.5, b=k ** 0.5)
            else:
                print(name)
                nn.init.uniform_(param.data, 0)


# train a given model, using a pytorch dataloader, optimizer, and scheduler (if provided)
def train(model,
          dataloader,
          optimizer,
          device,
          clip: float,
          scheduler=None):
    model.train()

    epoch_loss = 0

    for batch in dataloader:
        sentences, masks, labels, trig_mask, ent_mask = batch[0], batch[1], batch[2], batch[3], batch[4]

        optimizer.zero_grad()

        output = model(sentences.to(device), masks.to(device), trig_mask, ent_mask)
        loss = F.cross_entropy(output, labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    bert_model = AutoModel.from_pretrained(model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--re_model", type=str,
                        help="pass in the model type for relation extraction either r_bert or custom")
    parser.add_argument("-d", "--data_size", type=float,
                        help="pass the percentage of data to use")

    parser.add_argument("-t", "--train_type", type=str,
                        help="pass training environment for example 'colab'")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="pass batch size")

    parser.add_argument("-lr", "--lr_rate", type=float,
                        help="pass learning rate")

    parser.add_argument("-e", "--num_epochs", type=int,
                        help="pass num_epochs")
    args = parser.parse_args()
    re_model_name = args.re_model if args.re_model is not None else 'custom'
    batch_size = args.batch_size if args.batch_size is not None else 32
    data_amount = args.data_size if args.data_size is not None else 1
    N_EPOCHS = args.num_epochs if args.num_epochs is not None else 3
    LR = args.lr_rate if args.lr_rate is not None else 1e-5
    train_type = args.train_type
    print('hyper parameters')
    print(f'batch_size: {batch_size}')
    print(f'epochs: {N_EPOCHS}')
    print(f'LR: {LR}')
    print(f"using {data_amount * 100:.2f}% of training and validation data")
    print("loading data...")
    data_start = time.time()

    if train_type == 'colab':
        with open("data/ee_data.pickle", "rb") as f:
            data_brat = pickle.load(f)
        train_dataloader, val_dataloader, biobert_tokenizer = ee_preprocessor.colab_load_data(model_name, data_brat,
                                                                                              data_size=data_amount,
                                                                                              batch_size=batch_size)
    else:
        train_dataloader, val_dataloader, biobert_tokenizer = ee_preprocessor.load_data_ee(model_name)
    data_end = time.time()
    print(f'completed data loading in {(data_end - data_start) / 60:.2f} minutes')
    _, i2label = ee_preprocessor.map2ind()
    bert_model.resize_token_embeddings(len(biobert_tokenizer))  # adds 4 to account for relation tokens

    # define hyperparameters
    # BATCH_SIZE = 10
    # LR = 1e-5
    WEIGHT_DECAY = 0
    # N_EPOCHS = 3
    CLIP = 1.0

    # define models, move to device, and initialize weights
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    if re_model_name == 'r_bert':
        print("using r_bert model")
        model = R_bert.RelationClassifier(bert_model).to(device)
    else:
        print("using custom model")
        model = custom_model.RelationClassifier(bert_model).to(device)
    model.apply(init_classification_head_weights)
    model.to(device)
    print('Model Initialized')

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                num_training_steps=N_EPOCHS * len(train_dataloader))

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # print('running initial performance metrics')
    # start_time = time.time()
    # train_loss = evaluate(model, train_dataloader, device)
    # train_f1 = measure_f1(model, train_dataloader, device, i2label)

    # valid_loss = evaluate(model, val_dataloader, device)
    # valid_f1 = measure_f1(model, val_dataloader, device, i2label)

    # print(f'Initial Train Loss: {train_loss:.3f}')
    # print(f'Initial Train F1: {train_f1:.3f}')
    # print(f'Initial Valid Loss: {valid_loss:.3f}')
    # print(f'Initial Valid F1: {valid_f1:.3f}')
    # end_time = time.time()
    # print(f'completed initial performance metrics in {(end_time - start_time) / 60:.2f} minutes')

    print(f'training for {N_EPOCHS} epochs')
    train_start_time = time.time()
    for epoch in tqdm(range(N_EPOCHS)):
        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, device, CLIP, scheduler)
        end_time = time.time()
        train_f1 = measure_f1(model, train_dataloader, device, i2label)
        valid_loss = evaluate(model, val_dataloader, device)
        valid_f1 = measure_f1(model, val_dataloader, device, i2label)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTrain F1: {train_f1:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tValid F1: {valid_f1:.3f}')

    train_end_time = time.time()
    print(f'completed training in {(train_end_time - train_start_time) / 60:.2f} minutes')

    print('saving model')
    torch.save(model.state_dict(), f"src/re_models/{re_model_name}_batch{batch_size}_lr{LR}_epochs{N_EPOCHS}.pt")
    # print('test loading model')
    # loaded_model = RelationClassifier(bert_model).to(device)
    # loaded_model.load_state_dict(torch.load("results/re_custom_model.pt", map_location=device))