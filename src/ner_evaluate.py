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



# Set what version of model to load in
VERSION = 2
MODEL_TYPE = "Vanilla" # CRF was not saved and will not work for evaluation
# Task can be changed to 1 or 2
TASK = 1
# Loading in directory of saved model
MODEL_NAME = "ner_models/ner_task_" + str(TASK) + '_v' + str(VERSION) # "dslim/bert-base-NER"



TOKENIZER = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2" )

tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data(TASK, 16, "dmis-lab/biobert-base-cased-v1.2")

NUM_TAGS = max(list(i2tag_train.keys())) + 1

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


#####################################################################
# Vanilla
#####################################################################

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
results, label_tags = validate()
f1_micro, f1_macro, precision, recall = measure_results(results, label_tags)
