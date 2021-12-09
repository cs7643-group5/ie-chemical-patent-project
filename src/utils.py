from preprocessing import ee_preprocessor
import ee_evaluation
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM
import time

from re_models import custom_model


def store_data():

    sentences_train, tags_train, ee_sentences_train, ee_labels_train = ee_preprocessor.read_folder_ee('data/ee_train/')
    sentences_val, tags_val, ee_sentences_val, ee_labels_val = ee_preprocessor.read_folder_ee('data/ee_dev/')

    data_train = (ee_sentences_train, ee_labels_train)
    data_val = (ee_sentences_val, ee_labels_val)

    with open("data/ee_data.pickle", "wb") as f:
        pickle.dump((data_train, data_val), f)

    return None

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
    for label in i2label.values():
        if label != 'O':
            precision = round(results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fp'] + 1e-6), 4)
            recall = round(results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fn'] + 1e-6), 4)
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
    print('Average non-zero F1 Score: ', round(f1_average, 4))
    print("Overall Performance, Precision: ", precision_overall, ", Recall: ", recall_overall, ", F1: ", f1_overall)
    return None

def re_evaluate(model):

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')


    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data_size", type=float,
    #                     help="pass the percentage of data to use")
    #
    # parser.add_argument("-t", "--train_type", type=str,
    #                     help="pass training environment for example 'colab'")
    # args = parser.parse_args()
    # data_amount = args.data_size if args.data_size is not None else 1
    # train_type = args.train_type
    # print(f"using {data_amount * 100:.2f}% of training and validation data")
    print("loading data...")
    data_start = time.time()

    with open("data/ee_data.pickle", "rb") as f:
        data_brat = pickle.load(f)
    train_dataloader, val_dataloader, biobert_tokenizer = ee_preprocessor.colab_load_data(model_name, data_brat,
                                                                                          data_size=1)
    data_end = time.time()
    print(f'completed data loading in {(data_end - data_start) / 60:.2f} minutes')
    _, i2label = ee_preprocessor.map2ind()

    model_name = "dmis-lab/biobert-base-cased-v1.2"
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.resize_token_embeddings(len(biobert_tokenizer))  # adds 4 to account for relation tokens
    model_re = custom_model.RelationClassifier(bert_model).to(device)
    model_re.load_state_dict(torch.load("saved_models/re_custom_model.pt", map_location=device))

    measure_f1(model, val_dataloader, device, i2label)





#store_data()
# missed_entity_pairs = ee_evaluation.store_data()
# with open("data/missed_entities_pairs.pickle", "wb") as f:
#     pickle.dump(missed_entity_pairs, f)


