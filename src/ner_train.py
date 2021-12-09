# # if ie-chemical-patent-project is working directory
# # run following command to test out data loading
# # python src/preprocessing/run_preprocessor.py


# Import libraries
from preprocessing import preprocessor
from transformers import AdamW, AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification, AutoModelForTokenClassification, AutoModel
from transformers import AutoModelForTokenClassification, AutoModel, get_linear_schedule_with_warmup
import torch, itertools
from torchcrf import CRF
import torch.nn as nn
import numpy as np


# Torch Device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

print("Device: ", device)


# Load Data Arguments
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2" # "dslim/bert-base-NER"
MODEL_TYPE = "Vanilla"
TASK = 1
VERSION = 4

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL_UPLOAD_NAME = 'ner_task_' + str(TASK)

BASELINE_BATCH_SIZE = 32
BASELINE_LR = 5e-5

BATCH_SIZE = 16
# Learning Rate is adjusted based on batch size, increasing by the root of batch size.
LR = BASELINE_LR * np.sqrt(BATCH_SIZE / BASELINE_BATCH_SIZE)
EPOCHS = 50

tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data(TASK, BATCH_SIZE, MODEL_NAME)

NUM_TAGS = max(list(i2tag_train.keys())) + 1
DROPOUT = 0.1
scheduler = None

best_model = None
best_f1 = 0

print()
print("Batch Size: ", BATCH_SIZE)
print("Learning Rate: ", LR)
print("Epochs: ", EPOCHS)
print("Number of Tags: ", NUM_TAGS)
print()



# Universal function to compare predicted tags and actual tags
def measure_results(results, label_tags):
    pred_tags = [item for sublist1 in results for sublist0 in sublist1 for item in sublist0]
    true_tags = [item for sublist1 in label_tags for sublist0 in sublist1 for item in sublist0]

    # classification_report([[item for item in sublist0] for sublist1 in results for sublist0 in sublist1], [[item for item in sublist0] for sublist1 in results for sublist0 in sublist1])
    f1_scores = []
    precision_scores = []
    recall_scores = []
    tags_count = []

    for tag in i2tag_train.values():

        if str(tag).lower().strip() in ['sep', 'cls', 'm']:
            continue
        
        tp_overall, fp_overall, fn_overall, tp, fp, fn = 0, 0, 0, 0, 0, 0

        for result, label in zip(pred_tags, true_tags):
            if result == tag and label == tag:
                tp_overall += 1
                tp += 1
            if result == tag and label != tag:
                fp_overall += 1
                fp += 1
            if result != tag and label == tag:
                fn_overall += 1
                fn += 1
        
        precision = round(tp / (tp + fp + 1e-6), 4)
        recall = round(tp / (tp + fn + 1e-6), 4)
        f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)

        f1_scores.append(f1)

        count = tp + fn
        tags_count.append(count)

        print("Tag: ", tag, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)
    
    f1_micro_average = sum(f1_scores)/len(f1_scores)
    precision_average = round(tp_overall / (tp_overall + fp_overall + 1e-6), 6)
    recall_average = round(tp_overall / (tp_overall + fn_overall + 1e-6), 6)
    f1_macro_average = round((2 * precision_average * recall_average) / (precision_average + recall_average + 1e-6), 6)

    print()
    print('Average F1 Micro Score: ', round(f1_micro_average, 4))
    print('Precision Score: ', round(precision_average, 6))
    print('Recall Score: ', round(recall_average, 6))
    print('Average F1 Macro Score: ', round(f1_macro_average, 6))

    return f1_micro_average, f1_macro_average,precision_average, recall_average 


# Vanilla Inference - Getting predictions and concatenating them along with labels in lists of lists
def validate():
    model.eval()
    
    results_full = []
    labels_full = []

    for input_ids, mask, labels in val_dataloader:
        outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
        outputs = outputs.logits[:, :, :NUM_TAGS].argmax(dim = -1)
        a, b = outputs.shape
        
        results = [[i2tag_val[outputs[i, j].item()] for j in range(b)] for i in range(a)]
        label_tags = [[i2tag_val[labels[i, j].item()] for j in range(b)] for i in range(a)] 

        results_full.append(results)
        labels_full.append(label_tags)
        
    return results_full, labels_full


# Same as validate funtion except customized for CRF
def crf_decoder():
    
    results_full = []
    labels_full = []

    for input_ids, mask, labels in val_dataloader:
        emissions = model.return_emissions(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
        outputs = model.crf.decode(emissions)
        a, b = len(outputs), len(outputs[0])
        
        results = [[i2tag_val[outputs[i][j]] for j in range(b)] for i in range(a)]
        label_tags = [[i2tag_val[labels[i][j].item()] for j in range(b)] for i in range(a)] 
        
        results_full.append(results)
        labels_full.append(label_tags)

    return results_full, labels_full




if MODEL_TYPE == 'Vanilla':

    #####################################################################
    # Vanilla
    #####################################################################

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

    optim = AdamW(model.parameters(), lr = LR)
    scheduler = get_linear_schedule_with_warmup(optim, len(train_dataloader), len(train_dataloader)*EPOCHS, last_epoch = -1 )

    # BASELINE TRAINING SCRIPT
    for epoch in range(EPOCHS):
        print()
        print("Epoch: ", epoch)
        print('LR: ', optim.param_groups[0]["lr"])

        model.train()
        for input_ids, mask, labels in train_dataloader:
            optim.zero_grad()
            outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
            loss = outputs.loss
            loss.backward()
            optim.step()
            if scheduler != None:
              scheduler.step()
        
        print(f"loss on epoch {epoch} = {loss}")
        results, label_tags = validate()
        f1_micro, f1_macro, precision, recall = measure_results(results, label_tags)

        if f1_micro > best_f1:
          best_f1 = f1_micro
          best_scores = [f1_micro, f1_macro, precision, recall]
          save_dir = "ner_models/" + MODEL_UPLOAD_NAME + '_' + str(VERSION)
        
        print('Best Scores (F1 Micro, F1 Macro, Precision, Recall): ', best_scores)
        model.save_pretrained(save_dir, push_to_hub=False)



elif MODEL_TYPE == 'CRF':

    ######################################################################
    # CRF
    ######################################################################

    # CRF TRAINING SCRIPT
    class CustomBERTModel(nn.Module):
        def __init__(self, num_tags = 15):
            
            super(CustomBERTModel, self).__init__()
            
            self.bert = AutoModelForMaskedLM.from_pretrained("model")#MODEL_NAME
            self.crf = CRF(num_tags)#batch_first = True

        def return_emissions(self, input_ids, attention_mask, labels, inference = True):
            
            if inference == True:
                self.bert.eval()

            outputs = self.bert(input_ids.to(device), attention_mask = attention_mask.to(device), labels=labels.to(device))
            emissions = torch.softmax(outputs.logits[:, :, :NUM_TAGS], dim = -1)
            
            emissions = torch.transpose(emissions, 0, 1)
            return emissions

        def forward(self, input_ids, mask, labels):
            
            outputs = self.bert(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
            emissions = torch.softmax(outputs.logits[:, :, :NUM_TAGS], dim = -1)
            
            emissions = torch.transpose(emissions, 0, 1)
            labels = torch.transpose(labels, 0, 1)
            
            loss = self.crf(emissions.to(device), labels.to(device))
            return -loss

    model = CustomBERTModel(num_tags = NUM_TAGS).to(torch.device(device)) ## can be gpu

    # optim = AdamW(model.parameters(), lr=LR)
    # optim = AdamW(model.crf.parameters(), lr=1e-4)
    optim = AdamW(
        [
            {'params': model.base.parameters()}, 
            {"params": model.crf.parameters(), "lr": 5e-5},
        ],
        lr=5e-6,
    )

    scheduler = get_linear_schedule_with_warmup(optim, len(train_dataloader), len(train_dataloader)*EPOCHS, last_epoch = -1 )

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        
        for input_ids, mask, labels in train_dataloader:
            optim.zero_grad()
            loss = model.forward(input_ids, mask, labels)
            loss.backward()
            optim.step()
            if scheduler != None:
              scheduler.step()

        print(f"loss on epoch {epoch} = {loss}")

        results, label_tags = crf_decoder()
        f1_micro, f1_macro, precision, recall = measure_results(results, label_tags)

        if f1_micro > best_f1:
          best_f1 = f1_micro
          best_scores = [f1_micro, f1_macro, precision, recall]

          save_dir = "ner_models/" + MODEL_UPLOAD_NAME + '_' + str(VERSION)
          model.save_pretrained(save_dir, push_to_hub=False)
        
        print('Best Scores (F1 Micro, F1 Macro, Precision, Recall): ', best_scores)


