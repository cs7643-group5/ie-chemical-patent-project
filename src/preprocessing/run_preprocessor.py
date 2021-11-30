# # if ie-chemical-patent-project is working directory
# # run following command to test out data loading
# # python src/preprocessing/run_preprocessor.py


# for batch_sents, batch_masks, batch_labels in train_dataloader:
#     print(f'\nbatch sents shape: {batch_sents.shape}')
#     print(f'batch masks shape: {batch_masks.shape}')
#     print(f'batch labels shape: {batch_labels.shape}\n')



# Import libraries
import preprocessor
from transformers import AdamW, AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
import torch, itertools
from conlleval import evaluate
from torchcrf import CRF
import torch.nn as nn

# Torch Device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
print("Device: ", device)



# Initializations
tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data()
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_tags = max(list(i2tag_train.keys())) + 1
dropout = 0.1



# Universal function to compare predicted tags and actual tags
def measure_results(results, label_tags):
    true_tags = [item for sublist1 in results for sublist0 in sublist1 for item in sublist0]
    pred_tags = [item for sublist1 in label_tags for sublist0 in sublist1 for item in sublist0]

    for tag in i2tag_train.values():
        
        tp, fp, fn = 0, 0, 0

        for result, label in zip(true_tags, pred_tags):
            if result == tag and label == tag:
                tp += 1
            if result == tag and label != tag:
                fp += 1
            if result != tag and label == tag:
                fn += 1
        
        precision = round(tp / (tp + fp + 1e-6), 4)
        recall = round(tp / (tp + fn + 1e-6), 4)
        f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)

        print("Tag: ", tag, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)






######################################################################
# CRF
######################################################################

# Iterating over the validation_dataloader and getting the emissions. Using the emissions, we run crf.decode
# def crf_decoder():
    
#     results_full = []
#     labels_full = []

#     for input_ids, mask, labels in val_dataloader:
#         emmissions = model.return_emissions(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
#         outputs = model.crf.decode(emissions)
#         a, b = len(outputs), len(outputs[0])
        
#         results = [[i2tag_val[outputs[i][j]] for j in range(b)] for i in range(a)]
#         label_tags = [[i2tag_val[labels[i][j]] for j in range(b)] for i in range(a)] 
        
#         results_full.append(results)
#         labels_full.append(label_tags)

#     return results_full, labels_full



# # CRF TRAINING SCRIPT
# class CustomBERTModel(nn.Module):
#     def __init__(self, num_tags = 12):
#         super(CustomBERTModel, self).__init__()
#         self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
#         self.linear1 = nn.Linear(28996, num_tags)
#         self.crf = CRF(num_tags+ 1)

#     def return_emissions(self, input_ids, mask, label, inference = True):
#         if inference == True:
#             self.bert.eval()
#         outputs = self.bert(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
#         emissions = self.linear1(outputs.logits)
#         return emissions

#     def compute_loss(self, input_ids, mask, labels):
#         outputs = self.bert(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
#         emissions = self.linear1(outputs.logits)
#         print(emissions.shape)
#         loss = self.crf(emissions.to(device), labels.to(device))
#         return -loss

# model = CustomBERTModel(num_tags = 12) # You can pass the parameters if required to have more flexible model
# model.to(torch.device(device)) ## can be gpu

# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     print("Epoch: ", epoch)

#     for input_ids, mask, labels in train_dataloader:
#         optim.zero_grad()
#         loss = model.compute_loss(input_ids, mask, labels)
#         loss.backward()
#         optim.step()
    
#     print(f"loss on epoch {epoch} = {loss}")
    
#     results, label_tags = crf_decoder()
#     measure_results(results, label_tags)










######################################################################
# Vanilla
######################################################################

# model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_tags).to(device)
# print(model)
# optim = AdamW(model.parameters(), lr=5e-5)


# def validate():
#     model.eval()
    
#     results_full = []
#     labels_full = []

#     for input_ids, mask, labels in val_dataloader:
#         outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
#         outputs = outputs.logits.argmax(dim = -1)
#         a, b = outputs.shape
        
#         results = [[i2tag_val[outputs[i, j].item()] for j in range(b)] for i in range(a)]
#         label_tags = [[i2tag_val[labels[i, j].item()] for j in range(b)] for i in range(a)] 
        
#         results_full.append(results)
#         labels_full.append(label_tags)

#     return results_full, labels_full


# # BASELINE TRAINING SCRIPT
# for epoch in range(3):
#     print("Epoch: ", epoch)

#     model.train()
#     for input_ids, mask, labels in train_dataloader:
#         optim.zero_grad()
#         outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
#         loss = outputs.loss
#         loss.backward()
#         optim.step()
    
#     print(f"loss on epoch {epoch} = {loss}")
#     results, label_tags = validate()
#     measure_results(results, label_tags)












######################################################################
# Vanilla w/ Classification Layer
######################################################################

def validate():
    model.eval()
    
    results_full = []
    labels_full = []

    for input_ids, mask, labels in val_dataloader:
        outputs = model.forward(input_ids, mask, labels)
        outputs = outputs.argmax(dim = -1)
        a, b = outputs.shape
        
        results = [[i2tag_val[outputs[i, j].item()] for j in range(b)] for i in range(a)]
        label_tags = [[i2tag_val[labels[i, j].item()] for j in range(b)] for i in range(a)] 
        
        results_full.append(results)
        labels_full.append(label_tags)

    return results_full, labels_full


class CustomBERTModel(nn.Module):
    def __init__(self, num_tags = num_tags):
        super(CustomBERTModel, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.linear1 = nn.Linear(28996, num_tags, dropout = dropout)

    def forward(self, input_ids, mask, labels):
        outputs = self.bert(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
        outputs = self.linear1(outputs.logits)
        return outputs


model = CustomBERTModel(num_tags = num_tags)
model.to(torch.device(device))
optim = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss() ## If required define your own criterion

# # BASELINE TRAINING SCRIPT w/ modifications
for epoch in range(10):
    print("Epoch: ", epoch)

    model.train()
    for input_ids, mask, labels in train_dataloader:
        optim.zero_grad()
        outputs = model.forward(input_ids, mask, labels)
        N, D, C = outputs.shape
        outputs = outputs.view(N*D, C)
        labels = labels.view(N*D)
        loss = criterion(outputs.cuda(), labels.cuda())
        loss.backward()
        optim.step()
    
    print(f"loss on epoch {epoch} = {loss}")
    results, label_tags = validate()
    measure_results(results, label_tags)


    



















# GENERAL DEBUGGING AND STUFF

# model.eval()
# print("OUTPUT LOGITS SHAPE: ", outputs.logits.shape)
# outputs = outputs.logits.argmax(dim = 2)
# a, b = outputs.shape

# print('RESULTS: ')
# results = [[i2tag_train[outputs[i, j].item()] for j in range(b)] for i in range(a)]
# print(results) 

# print("LABELS: ")
# label_tags = [[i2tag_train[labels[i, j].item()] for j in range(b)] for i in range(a)] 
# print(label_tags)

# true_tags = list(itertools.chain(*label_tags))
# pred_tags = list(itertools.chain(*results))

# for tag in i2tag_train.values():
#     tp = 0
#     fp = 0
#     fn = 0
#     for result, label in zip(true_tags, pred_tags):
#         if result == tag and label == tag:
#             tp += 1
#         if result == tag and label != tag:
#             fp += 1
#         if result != tag and label == tag:
#             fn += 1

#     precision = round(tp / (tp + fp + 1e-6))
#     recall = round(tp / (tp + fn + 1e-6))
#     f1 = round((2 * precision * recall) / (precision + recall + 1e-6))
    
#     print("Tag: ", tag, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

# true_tags = label_tags#list(itertools.chain(*label_tags))
# pred_tags = results#list(itertools.chain(*results))
# # print out the table as above
# evaluate(true_tags, pred_tags, verbose=True) 

# # calculate overall metrics
# prec, rec, f1 = evaluate(true_tags, pred_tags, verbose=False)

# calculate overall metrics
# prec, rec, f1 = evaluate(true_tags, pred_tags, verbose=False)

# def write_predictions(sentences, outFile):
#   fOut = open(outFile, 'w')
#   for y in sentences:
#     fOut.write("\n".join(y[1:len(y)-1]))  #Skip start and end tokens
#     fOut.write("\n\n")

# write_predictions(results, 'results')
# write_predictions(labels, 'labels')   #Performance on dev set
# print('conlleval:')
# print(subprocess.Popen('paste dev dev_pred | perl conlleval.pl -d "\t"', shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('UTF-8'))


    # def print_predictions(self, words, tags):
    #   Y_pred = self.inference(words)
    #   for i in range(len(words)):
    #     print("----------------------------")
    #     print(" ".join([f"{words[i][j]}/{Y_pred[i][j]}/{tags[i][j]}" for j in range(len(words[i]))]))
    #     print("Predicted:\t", Y_pred[i])
    #     print("Gold:\t\t", tags[i])


    # print(f"loss on epoch {epoch} = {totalLoss}")
# print('\nexample training')
# print('sentence')
# print(train_data[0][-1])
# print('sentence tags')
# print(train_data[1][-1])
#
# print('\nexample validation')
# print('sentence')
# print(val_data[0][-1])
# print('sentence tags')
# print(val_data[1][-1])


