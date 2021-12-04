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

from re_models.custom_model import RelationClassifier
from preprocessing import ee_preprocessor
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# count the number of trainable parameters in the model
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    train_dataloader, val_dataloader, biobert_tokenizer = ee_preprocessor.load_data_ee(model_name)
    bert_model.resize_token_embeddings(len(biobert_tokenizer))  # adds 4 to account for relation tokens

    # define hyperparameters
    BATCH_SIZE = 10
    LR = 1e-5
    WEIGHT_DECAY = 0
    N_EPOCHS = 3
    CLIP = 1.0

    # define models, move to device, and initialize weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RelationClassifier(bert_model).to(device)
    model.apply(init_classification_head_weights)
    model.to(device)
    print('Model Initialized')

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=N_EPOCHS*len(train_dataloader))

    print(f'The model has {count_parameters(model):,} trainable parameters')

    train_loss = evaluate(model, train_dataloader, device)
    train_acc = evaluate_acc(model, train_dataloader, device)

    valid_loss = evaluate(model, val_dataloader, device)
    valid_acc = evaluate_acc(model, val_dataloader, device)

    print(f'Initial Train Loss: {train_loss:.3f}')
    print(f'Initial Train Acc: {train_acc:.3f}')
    print(f'Initial Valid Loss: {valid_loss:.3f}')
    print(f'Initial Valid Acc: {valid_acc:.3f}')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, device, CLIP, scheduler)
        end_time = time.time()
        train_acc = evaluate_acc(model, train_dataloader, device)
        valid_loss = evaluate(model, val_dataloader, device)
        valid_acc = evaluate_acc(model, val_dataloader, device)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTrain Acc: {train_acc:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tValid Acc: {valid_acc:.3f}')