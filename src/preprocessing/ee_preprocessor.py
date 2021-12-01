
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


def def_value():
    return "O"


def create_re_examples(tag2wid_fixed, ids2arg,
                       sentences,
                       trigger_set, entity_set,
                       previous_tagid, w, z,
                       ee_sentences_fixed, ee_labels_fixed):

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

            ent_start = tag2wid[entity_id + 'start']
            ent_end = tag2wid[entity_id + 'end'] + 1  # add one to adjust for first insert of start

            # account for shift in the indices from the trigger inserts
            if ent_start > trig_start:
                ent_start += 2
                ent_end += 2

            adjust_sent.insert(trig_start, '$')
            adjust_sent.insert(trig_end, '$')
            adjust_sent.insert(ent_start, '#')
            adjust_sent.insert(ent_end, '#')

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


def load_data_ee():
    sentences_train, tags_train, ee_sentences_train, ee_labels_train = read_folder_ee('data/ee_train/')
    return sentences_train, tags_train, ee_sentences_train, ee_labels_train