# import preprocessor

# # if ie-chemical-patent-project is working directory
# # run following command to test out data loading
# # python src/preprocessing/run_preprocessor.py

# train_dataloader, val_dataloader = preprocessor.load_data()

# for batch_sents, batch_masks, batch_labels in train_dataloader:
#     print(f'\nbatch sents shape: {batch_sents.shape}')
#     print(f'batch masks shape: {batch_masks.shape}')
#     print(f'batch labels shape: {batch_labels.shape}\n')

import preprocessor
from transformers import AdamW, AutoTokenizer, AutoModelForMaskedLM
import torch, itertools
from conlleval import evaluate


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
print("Device: ", device)


tag2i_train, i2tag_train, tag2i_val, i2tag_val, train_dataloader, val_dataloader = preprocessor.load_data()

model_name = "dmis-lab/biobert-base-cased-v1.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
optim = AdamW(model.parameters(), lr=5e-5)
def validate(datashape):
    model.eval()
    
    results_full = []
    labels_full = []

    for input_ids, mask, labels in val_dataloader:
        outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
        outputs = outputs.logits.argmax(dim = -1)
        a, b = outputs.shape
        
        results = [[i2tag_val[outputs[i, j].item()] for j in range(b)] for i in range(a)]
        label_tags = [[i2tag_val[labels[i, j].item()] for j in range(b)] for i in range(a)] 
        
        results_full.append(results)
        labels_full.append(label_tags)

    return results_full, labels_full



def measure_results(results, label_tags):
    true_tags = [item for sublist1 in results for sublist0 in sublist1 for item in sublist0]
    pred_tags = [item for sublist1 in label_tags for sublist0 in sublist1 for item in sublist0]

    for tag in i2tag_train.values():
        tp = 0
        fp = 0
        fn = 0
        for result, label in zip(true_tags, pred_tags):
            if result == tag and label == tag:
                tp += 1
            if result == tag and label != tag:
                fp += 1
            if result != tag and label == tag:
                fn += 1
        
        precision = round(tp / (tp + fp + 1e-6))
        recall = round(tp / (tp + fn + 1e-6))
        f1 = round((2 * precision * recall) / (precision + recall + 1e-6))

        print("Tag: ", tag, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)



for epoch in range(3):
    print("Epoch: ", epoch)

    model.train()
    for input_ids, mask, labels in train_dataloader:
        optim.zero_grad()
        outputs = model(input_ids.to(device), attention_mask = mask.to(device), labels=labels.to(device))
        loss = outputs.loss
        loss.backward()
        optim.step()
    
    print(f"loss on epoch {epoch} = {loss}")
    results, label_tags = validate(input_ids.shape)
    measure_results(results, label_tags)

model.eval()
outputs = outputs.logits.argmax(dim = 2)
a, b = outputs.shape

print('RESULTS: ')
results = [[i2tag_train[outputs[i, j].item()] for j in range(b)] for i in range(a)]
print(results) 

print("LABELS: ")
label_tags = [[i2tag_train[labels[i, j].item()] for j in range(b)] for i in range(a)] 
print(label_tags)



true_tags = list(itertools.chain(*label_tags))
pred_tags = list(itertools.chain(*results))

for tag in i2tag_train.values():
    tp = 0
    fp = 0
    fn = 0
    for result, label in zip(true_tags, pred_tags):
        if result == tag and label == tag:
            tp += 1
        if result == tag and label != tag:
            fp += 1
        if result != tag and label == tag:
            fn += 1

    precision = round(tp / (tp + fp + 1e-6))
    recall = round(tp / (tp + fn + 1e-6))
    f1 = round((2 * precision * recall) / (precision + recall + 1e-6))
    
    print("Tag: ", tag, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

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


