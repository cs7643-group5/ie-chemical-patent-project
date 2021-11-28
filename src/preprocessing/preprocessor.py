from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

import pdb


from functools import partial
from transformers import AutoTokenizer, AutoModelForMaskedLM


import torch

import numpy as np

import re
import os

from io import open


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def read_brat_format(ann_filename, txt_filename):

    # read in raw text from patent snippet
    snippet = ""
    for line in open(txt_filename).readlines():
        snippet += line

    if ann_filename is not None:
        # create an array of tags with O
        char_tags = ['O']*len(snippet)

        # go through each line of the annotation file and place the tag over every character position in char_tags
        # this will repeat the tag multiple times which is inefficient but it's a good way to get the data in the right form
        for line in open(ann_filename).readlines():
            line_seg = line.split()
            tag = line_seg[1]
            start_i = int(line_seg[2])
            end_i = int(line_seg[3])
            char_tags[start_i:end_i] = [tag]*(end_i-start_i)

        z = 0  # place holder to determine what sentence index on currently
        sent_spans = list(PunktSentenceTokenizer().span_tokenize(snippet))  # get spans for sentences
        sentences = [[]] * len(sent_spans)  # create an empty list of lists for sentences
        tags = [[]] * len(sent_spans)  # create an empty list of lists for tags

        # go over each word in the snippet defined by white space separation
        for span_b, span_e in WhitespaceTokenizer().span_tokenize(snippet):
            # index out to get the word
            word = snippet[span_b:span_e]
            # make sure to get rid of leading and trailing non-alphanumeric characters
            word = re.sub(r'^\W+|\W+$', '', word)
            # the tag will be repeated in char_tags for char_tags[span_b:span_e], we only want one of them so can use span_b
            # to index
            tag = char_tags[span_b]

            # if a token corresponding to the next sequence is reached increase z and append END token
            if span_b >= sent_spans[z][1]:
                sentences[z].append('[SEP]')
                tags[z].append('SEP')
                z += 1

            #if the current sentence is empty place the START token
            if len(sentences[z]) == 0:
                sentences[z] = ['[CLS]']
                tags[z] = ['CLS']

            sentences[z].append(word)
            tags[z].append(tag)

        # last iterations ending tokens will be missing so have to add in here
        sentences[z].append('[SEP]')
        tags[z].append('SEP')

    else:
        ''' need a more efficient way to run through test snippets also not sure yet what format we need for test '''
        sentences = None
        tags = None
        # sent_spans = list(PunktSentenceTokenizer().span_tokenize(snippet))  # get spans for sentences
        # sentences = [[]] * len(sent_spans)  # create an empty list of lists for sentences
        # z = 0
        # tags = None
        # for span_b, span_e in WhitespaceTokenizer().span_tokenize(snippet):
        #     # index out to get the word
        #     word = snippet[span_b:span_e]
        #     # make sure to get rid of leading and trailing non-alphanumeric characters
        #     word = re.sub(r'^\W+|\W+$', '', word)
        #
        #     # if a token corresponding to the next sequence is reached increase z and append END token
        #     if span_b >= sent_spans[z][1]:
        #         sentences[z].append('-END-')
        #         z += 1
        #
        #     # if the current sentence is empty place the START token
        #     if len(sentences[z]) == 0:
        #         sentences[z] = ['-START-']
        #
        #     sentences[z].append(word)
        #
        # # last iterations ending tokens will be missing so have to add in here
        # sentences[z].append('-END-')
        #
        # # sentences = sent_tokenize(snippet)

    return sentences, tags


def read_folder(path, labels=True):

    files = sorted(os.listdir(path))
    sents_all = []
    tags_all = []
    if labels:
        tags_all = []
        for s_file, e_file in zip(files[:-1], files[1:]):
            if s_file[-3:] == 'ann' and e_file[-3:] == 'txt':
                s_id = re.findall(r'\d+', s_file)
                e_id = re.findall(r'\d+', e_file)
                if s_id[0] == e_id[0]:
                    sents, tags = read_brat_format(path+s_file, path+e_file)
                    sents_all.extend(sents)
                    tags_all.extend(tags)
    else:
        for f in files:
            sents, _ = read_brat_format(None, path + f)
            sents_all.extend(sents)

    return sents_all, tags_all


def map2ind(tags):
    tag2i = {t: i+2 for i, t in enumerate(set(tag for sent in tags for tag in sent))}
    i2tag = {i:t for t, i in tag2i.items()}
    return tag2i, i2tag


class PatentDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def transformer_collate_fn(batch, tokenizer, tag2id):

    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']

    sentences, mask, labels = [], [], []

    for data, target in batch:
        tokenized_label = [tag2id[tag] for tag in target]
        tokenized_sent = tokenizer.convert_tokens_to_ids(data)

        if len(tokenized_sent) != len(tokenized_label):
            raise Exception("target tags are not the same length as the input sequence")

        mask.append(torch.ones(len(tokenized_sent)))
        sentences.append(torch.tensor(tokenized_sent))
        labels.append(torch.tensor(tokenized_label))

    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    labels = pad_sequence(labels, batch_first=True, padding_value=bert_pad_token)
    mask = pad_sequence(mask, batch_first=True, padding_value=bert_pad_token)

    return sentences, mask, labels


def load_biobert(name):

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)

    return tokenizer, model


def load_data():

    train_sents, train_tags = read_folder('data/train/')
    val_sents, val_tags = read_folder('data/dev/')
    # test_sents, _ = read_folder('data/chemu.ner.test/', labels=False)

    model_name = "dmis-lab/biobert-base-cased-v1.2"

    biobert_tokenizer, biobert_model = load_biobert(model_name)

    tag2i_train, i2tag_train = map2ind(train_tags)
    tag2i_val, i2tag_val = map2ind(val_tags)

    train_dataset = PatentDataset(train_sents, train_tags)
    train_dataloader = DataLoader(train_dataset, batch_size=32,
                                  collate_fn=partial(transformer_collate_fn,
                                                     tokenizer=biobert_tokenizer,
                                                     tag2id=tag2i_train),
                                  shuffle=True)

    val_dataset = PatentDataset(val_sents, val_tags)
    val_dataloader = DataLoader(val_dataset, batch_size=32,
                                  collate_fn=partial(transformer_collate_fn,
                                                     tokenizer=biobert_tokenizer,
                                                     tag2id=tag2i_val),
                                  shuffle=True)

    return train_dataloader, val_dataloader


