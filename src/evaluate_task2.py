from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
import pickle
import argparse

from re_models import custom_model
from re_models import R_bert
import ee_evaluation
from preprocessing import ee_preprocessor
from transformers import AutoModel, AutoModelForMaskedLM

if __name__ == '__main__':
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    bert_model = AutoModel.from_pretrained(model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_size", type=float,
                        help="pass the percentage of data to use")

    # parser.add_argument("-t", "--train_type", type=str,
    #                     help="pass training environment for example 'colab' ")
    args = parser.parse_args()
    data_amount = args.data_size if args.data_size is not None else 1
    # train_type = args.train_type
    print(f"using {data_amount * 100:.2f}% of training and validation data")
    print("loading data...")
    data_start = time.time()

    # ensure that pickle file is placed in the specified path
    # if not please run store data from ee evaluation
    with open("data/ee_eval_data.pickle", "rb") as f:
        data_brat = pickle.load(f)
        i2tag, i2trigger, val_dataloader, biobert_tokenizer = ee_evaluation.colab_load_data(model_name, data_brat,
                                                                                            data_size=data_amount)

    data_end = time.time()

    print(f'completed data loading in {(data_end - data_start) / 60:.2f} minutes')

    # define models, move to device, and initialize weights
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    ####################################################################################################################
    # change the following lines to evaluate the respective models for NER and Event extraction for the task 2 pipeline
    ####################################################################################################################
    bert_model.resize_token_embeddings(len(biobert_tokenizer))
    # model_re = custom_model.RelationClassifier(bert_model).to(device)
    model_re = R_bert.RelationClassifier(bert_model).to(device)
    # model_re.load_state_dict(torch.load("src/re_models/re_custom_model.pt", map_location=device))
    model_re.load_state_dict(torch.load("src/re_models/r_bert_model_best.pt", map_location=device))

    model_ner = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model_ner.load_state_dict(torch.load("src/re_models/ner_task_2_v4/pytorch_model.bin", map_location=device))
    ####################################################################################################################
    ####################################################################################################################

    _, i2arg = ee_preprocessor.map2ind()

    ee_evaluation.measure_ee_f1(model_ner, model_re, val_dataloader, device, i2arg, i2trigger, i2tag, biobert_tokenizer)



