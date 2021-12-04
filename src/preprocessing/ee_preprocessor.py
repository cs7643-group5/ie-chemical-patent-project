
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

import pdb


from functools import partial
from transformers import AutoTokenizer, AutoModel


import torch

import numpy as np

import re
import os

from io import open


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import defaultdict


def def_value():
    return "O"


def create_re_examples(tag2wid_fixed, ids2arg,
                       sentences,
                       trigger_set, entity_set,
                       previous_tagid, w, z,
                       ee_sentences_fixed, ee_labels_fixed,
                       replace_entities=True):

    tag2wid = tag2wid_fixed.copy()
    ee_sentences = ee_sentences_fixed.copy()
    ee_labels = ee_labels_fixed.copy()

    # capture edge case where last word is an entity that needs an ending index
    # if tag_id != previous_tagid:
    if previous_tagid != 'O':
        # pdb.set_trace()
        if len(tag2wid) != 0:
            tag2wid[previous_tagid + 'end'] = w
    # pdb.set_trace()
    fixed_sent = sentences[z]
    for trigger, trigger_id in trigger_set:
        # pdb.set_trace()
        for entity, entity_id in entity_set:
            adjust_sent = fixed_sent.copy()
            trig_start = tag2wid[trigger_id + 'start']
            trig_end = tag2wid[trigger_id + 'end'] + 1  # add one to adjust for first insert of start

            adjust_sent.insert(trig_start, '[$]')
            adjust_sent.insert(trig_end, '[/$]')

            if replace_entities:
                trig_num_pops = trig_end-trig_start-1
                # alt method
                # for _ in range(trig_num_pops):
                #     adjust_sent.pop(trig_start+1)
                del adjust_sent[trig_start+1:trig_end]
                adjust_sent.insert(trig_start+1, trigger)

            ent_start = tag2wid[entity_id + 'start']
            ent_end = tag2wid[entity_id + 'end'] + 1  # add one to adjust for first insert of start

            # account for shift in the indices from the trigger inserts or deletes
            if ent_start > trig_start and not replace_entities:
                ent_start += 2
                ent_end += 2
            elif ent_start > trig_start and replace_entities:
                ent_start += 2 + 1 - trig_num_pops
                ent_end += 2 + 1 - trig_num_pops

            adjust_sent.insert(ent_start, '[#]')
            adjust_sent.insert(ent_end, '[/#]')

            if replace_entities:
                # ent_num_pops = ent_end - ent_start - 1
                # alt method
                # for _ in range(ent_num_pops):
                #     adjust_sent.pop(ent_start + 1)
                del adjust_sent[ent_start + 1:ent_end]
                adjust_sent.insert(ent_start + 1, entity)

            ee_sentences.append(adjust_sent)
            ee_labels.append(ids2arg[(trigger_id, entity_id)])

    return ee_sentences, ee_labels


def read_brat_format_ee(ann_filename, txt_filename):
    # read in raw text from patent snippet
    snippet = ""
    for line in open(txt_filename, encoding="utf8").readlines():
        snippet += line

    if ann_filename is not None:
        #pdb.set_trace()

        # create an array of tags with O
        char_tags = ['O']*len(snippet)
        id_tags = ['O']*len(snippet)
        ids2arg = defaultdict(def_value)

        # go through each line of the annotation file and place the tag over every character position in char_tags
        # this will repeat the tag multiple times which is inefficient but it's a good way to get the data in the right form
        for line in open(ann_filename, encoding="utf8").readlines():
            line_seg = line.split()
            # tag_id = line_seg[0]
            # tag_ids.append(tag_id)
            tag_id = line_seg[0]
            if tag_id[0] == 'T':
                tag = line_seg[1]
                start_i = int(line_seg[2])
                end_i = int(line_seg[3])
                char_tags[start_i:end_i] = [tag]*(end_i-start_i)
                id_tags[start_i:end_i] = [tag_id] * (end_i - start_i)
            elif tag_id[0] == 'R':
                #pdb.set_trace()
                #trigger, entity = re.findall('T\d\d', txt)
                trigger = line_seg[2][5:]
                entity = line_seg[3][5:]
                relation = line_seg[1] #re.findall('ARG\w', txt)
                # if len(relation) != 1:
                #     raise Exception("Debug here: relation size is greater than 1")
                ids2arg[(trigger, entity)] = relation

        #pdb.set_trace()

        z = 0  # place holder to determine what sentence index on currently
        w = 0  # place holder to determine what word index on currently
        sent_spans = list(PunktSentenceTokenizer().span_tokenize(snippet))  # get spans for sentences
        sentences = [[]] * len(sent_spans)  # create an empty list of lists for sentences
        ee_sentences = []
        ee_labels = []
        tags = [[]] * len(sent_spans)  # create an empty list of lists for tags
        tag2wid = {}
        previous_tagid = ''

        trigger_set = set()
        entity_set = set()

        # go over each word in the snippet defined by white space separation
        for span_b, span_e in WhitespaceTokenizer().span_tokenize(snippet):
            # pdb.set_trace()
            # index out to get the word
            word = snippet[span_b:span_e]
            # make sure to get rid of leading and trailing non-alphanumeric characters
            word = re.sub(r'^\W+|\W+$', '', word)
            # the tag will be repeated in char_tags for char_tags[span_b:span_e], we only want one of them so can use span_b
            # to index
            tag = char_tags[span_b]
            tag_id = id_tags[span_b]

            # if a token corresponding to the next sequence is reached increase z and append END token
            if span_b >= sent_spans[z][1]:
                sentences[z].append('[SEP]')
                tags[z].append('SEP')
                ee_sentences, ee_labels = create_re_examples(tag2wid, ids2arg,
                                                                              sentences,
                                                                              trigger_set, entity_set,
                                                                              previous_tagid, w, z,
                                                                              ee_sentences, ee_labels)

                # reset variables for next sentence
                trigger_set = set()
                entity_set = set()
                previous_tagid = ''
                tag2wid = {}

                z += 1
                w = 0

            #if the current sentence is empty place the START token
            if len(sentences[z]) == 0:
                sentences[z] = ['[CLS]']
                tags[z] = ['CLS']
                w += 1

            # need to identify if carry over from previous word
            if tag_id != previous_tagid:
                if len(tag2wid) != 0 and previous_tagid != 'O':
                    tag2wid[previous_tagid + 'end'] = w
                if tag_id != 'O':
                    tag2wid[tag_id + 'start'] = w
                previous_tagid = tag_id

            if tag == 'WORKUP' or tag == 'REACTION_STEP':
                # pdb.set_trace()
                trigger_set.add((tag, tag_id))
            elif tag != 'O':
                # pdb.set_trace()
                entity_set.add((tag, tag_id))

            sentences[z].append(word)
            tags[z].append(tag)

            w += 1

        # last iterations ending tokens will be missing so have to add in here
        sentences[z].append('[SEP]')
        tags[z].append('SEP')
        ee_sentences, ee_labels = create_re_examples(tag2wid, ids2arg,
                                                                      sentences,
                                                                      trigger_set, entity_set,
                                                                      previous_tagid, w, z,
                                                                      ee_sentences, ee_labels)

    return sentences, tags, ee_sentences, ee_labels


def read_folder_ee(path):
    files = sorted(os.listdir(path))
    sents_all = []
    tags_all = []
    sents_ee_all = []
    labels_ee_all = []

    for s_file, e_file in zip(files[:-1], files[1:]):
        if s_file[-3:] == 'ann' and e_file[-3:] == 'txt':
            s_id = re.findall(r'\d+', s_file)
            e_id = re.findall(r'\d+', e_file)
            if s_id[0] == e_id[0]:
                sents, tags, ee_sents, ee_labels = read_brat_format_ee(path+s_file, path+e_file)
                sents_all.extend(sents)
                tags_all.extend(tags)
                sents_ee_all.extend(ee_sents)
                labels_ee_all.extend(ee_labels)

    return sents_all, tags_all, sents_ee_all, labels_ee_all


# def map2ind(tags):
#     #tag2i = {t: i+2 for i, t in enumerate(set(tag for sent in tags for tag in sent))}
#     tag2i = {}
#     tag2i['O'] = 0
#     tag2i['ARGM'] = 1
#     tag2i['ARG1'] = 2
#     i2tag = {i:t for t, i in tag2i.items()}
#
#     return tag2i, i2tag


class PatentDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def transformer_collate_fn(batch, tokenizer, label2id, use_wordpiece=False):

    # pdb.set_trace()
    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    trig_start_token = bert_vocab['[$]']
    trig_end_token = bert_vocab['[/$]']
    ent_start_token = bert_vocab['[#]']
    ent_end_token = bert_vocab['[/#]']
    unk_token = bert_vocab['[UNK]']

    sentences, mask, labels, trig_mask, ent_mask = [], [], [], [], []

    for data, target in batch:

        tokenized_label = label2id[target]

        if use_wordpiece:
            tokenized = tokenizer(data, is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  add_special_tokens=False)
            tokenized_sent = tokenized.input_ids
        else:
            tokenized_sent = tokenizer.convert_tokens_to_ids(data)

        if len(tokenized_sent) == 512:
            raise Exception("maximum sentence length reached. sentence may have been truncated and could be missing"
                            "relation tags")

        mask.append(torch.ones(len(tokenized_sent)))
        sentences.append(torch.tensor(tokenized_sent))
        labels.append(torch.tensor(tokenized_label))
        trig_mask.append((tokenized_sent.index(trig_start_token),
                         tokenized_sent.index(trig_end_token)))
        ent_mask.append((tokenized_sent.index(ent_start_token),
                         tokenized_sent.index(ent_end_token)))

    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    mask = pad_sequence(mask, batch_first=True, padding_value=bert_pad_token)

    return sentences, mask, torch.tensor(labels), trig_mask, ent_mask


# def load_biobert(name):
#
#     tokenizer = AutoTokenizer.from_pretrained(name)
#     model = AutoModel.from_pretrained(name)
#
#     return tokenizer, model

def load_data_ee(model_name):
    sentences_train, tags_train, ee_sentences_train, ee_labels_train = read_folder_ee('data/ee_train/')
    sentences_val, tags_val, ee_sentences_val, ee_labels_val = read_folder_ee('data/ee_train/')

    train_dataset_re = PatentDataset(ee_sentences_train, ee_labels_train)
    val_dataset_re = PatentDataset(ee_sentences_val, ee_labels_val)

    biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add relation tokens
    biobert_tokenizer.add_special_tokens({'additional_special_tokens': ['[$]', '[/$]', '[#]', '[/#]']})

    label2i = dict()
    label2i['O'] = 0
    label2i['ARGM'] = 1
    label2i['ARG1'] = 2
    #i2label = {i:t for t, i in label2i.items()}

    train_dataloader = DataLoader(train_dataset_re, batch_size=32,
                                  collate_fn=partial(transformer_collate_fn,
                                                     tokenizer=biobert_tokenizer,
                                                     label2id=label2i,
                                                     use_wordpiece=False),
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset_re, batch_size=32,
                                  collate_fn=partial(transformer_collate_fn,
                                                     tokenizer=biobert_tokenizer,
                                                     label2id=label2i,
                                                     use_wordpiece=False),
                                  shuffle=True)



    # return sentences_train, tags_train, ee_sentences_train, ee_labels_train
    return train_dataloader, val_dataloader, biobert_tokenizer