

from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

from tqdm import tqdm

import time

import pdb

from functools import partial
from transformers import AutoTokenizer, AutoModel

import torch

from preprocessing import preprocessor

import numpy as np

import re
import os
import pickle

from io import open

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import defaultdict


def def_value():
    return "O"

def def_value_missed_pairs():
    return 0


def create_re_examples(tag2wid_fixed, ids2arg,
                       sentences,
                       trigger_set, entity_set,
                       previous_tagid, w, z,
                       ee_sentences_fixed, ee_labels_fixed,
                       ids2arg_sent_fixed,
                       snippet_entity_pairs_captured_fixed,
                       replace_entities=True):

    snippet_entity_pairs_captured = snippet_entity_pairs_captured_fixed.copy()
    ids2arg_sent = ids2arg_sent_fixed.copy()
    # ids2arg_per_sent = ids2arg_per_sent_fixed.copy()
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
                trig_num_pops = trig_end - trig_start - 1
                # alt method
                # for _ in range(trig_num_pops):
                #     adjust_sent.pop(trig_start+1)
                del adjust_sent[trig_start + 1:trig_end]
                adjust_sent.insert(trig_start + 1, trigger)

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
            # pdb.set_trace()
            if len(ids2arg_sent[z]) == 0:
                ids2arg_sent[z] = ({(trigger_id, entity_id): ids2arg[(trigger_id, entity_id)]}, {(trigger_id, entity_id): (trigger, entity)})
            else:
                ids2arg_sent[z][0][(trigger_id, entity_id)] = ids2arg[(trigger_id, entity_id)]
                ids2arg_sent[z][1][(trigger_id, entity_id)] = (trigger, entity)

    if len(ids2arg_sent[z]) == 0:
        ids2arg_sent[z] = ({}, {})

    if ids2arg_sent[z][0].keys() != ids2arg_sent[z][1].keys():
        pdb.set_trace()
        raise Exception('ids2arg tuple do not contain the same keys')

    for key in ids2arg_sent[z][0].keys():
        snippet_entity_pairs_captured.add(key)

    return ee_sentences, ee_labels, tag2wid, ids2arg_sent, snippet_entity_pairs_captured


def read_brat_format_ee(ann_filename, txt_filename, missed_entity_pairs_fixed):
    # read in raw text from patent snippet
    missed_entity_pairs = missed_entity_pairs_fixed.copy()
    snippet_entity_pairs_captured = set()
    snippet = ""
    for line in open(txt_filename, encoding="utf8").readlines():
        snippet += line

    if ann_filename is not None:
        # pdb.set_trace()

        # create an array of tags with O
        char_tags = ['O'] * len(snippet)
        id_tags = ['O'] * len(snippet)
        ids2arg = defaultdict(def_value)
        snippet_id_2_label = {}

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
                snippet_id_2_label[tag_id] = tag
                char_tags[start_i:end_i] = [tag] * (end_i - start_i)
                id_tags[start_i:end_i] = [tag_id] * (end_i - start_i)
            elif tag_id[0] == 'R':
                # pdb.set_trace()
                # trigger, entity = re.findall('T\d\d', txt)
                trigger = line_seg[2][5:]
                entity = line_seg[3][5:]
                relation = line_seg[1]  # re.findall('ARG\w', txt)
                # if len(relation) != 1:
                #     raise Exception("Debug here: relation size is greater than 1")
                ids2arg[(trigger, entity)] = relation

        # pdb.set_trace()

        z = 0  # place holder to determine what sentence index on currently
        w = 0  # place holder to determine what word index on currently
        sent_spans = list(PunktSentenceTokenizer().span_tokenize(snippet))  # get spans for sentences
        sentences = [[]] * len(sent_spans)  # create an empty list of lists for sentences
        ee_sentences = []
        ee_labels = []
        tags = [[]] * len(sent_spans)  # create an empty list of lists for tags
        tag2wid = {}
        previous_tagid = ''
        ee_ent_tag_ids = [[]] * len(sent_spans)
        ent_sets = [[]] * len(sent_spans)
        tag2wids = [[]] * len(sent_spans)
        # ids2arg_per_sent = [{}] * len(sentences)
        # ids2arg_name_per_sent = [{}] * len(sentences)
        ids2arg_sent = [[]] * len(sentences)

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
                ee_ent_tag_ids[z].append(('O', 'O'))
                if len(ee_ent_tag_ids[z]) != len(sentences[z]):
                    raise Exception('nope')
                ent_sets[z] = entity_set
                ee_sentences, ee_labels, tag2wid, ids2arg_sent, snippet_entity_pairs_captured = create_re_examples(tag2wid, ids2arg,
                                                             sentences,
                                                             trigger_set, entity_set,
                                                             previous_tagid, w, z,
                                                             ee_sentences, ee_labels,
                                                            ids2arg_sent, snippet_entity_pairs_captured)
                tag2wids[z] = tag2wid

                if len(ids2arg_sent[z]) != 2:
                    pdb.set_trace()
                    raise Exception('tuple size for ids and args are not equal to 2')

                # reset variables for next sentence
                trigger_set = set()
                entity_set = set()
                previous_tagid = ''
                tag2wid = {}

                z += 1
                w = 0

            # if the current sentence is empty place the START token
            if len(sentences[z]) == 0:
                #pdb.set_trace()
                sentences[z] = ['[CLS]']
                tags[z] = ['CLS']
                ee_ent_tag_ids[z] = [('O', 'O')]
                if len(ee_ent_tag_ids[z]) != len(sentences[z]):
                    raise Exception('nope')
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
            # pdb.set_trace()
            ee_ent_tag_ids[z].append((tag, tag_id))
            if len(ee_ent_tag_ids[z]) != len(sentences[z]):
                raise Exception('nope')

            w += 1

        # last iterations ending tokens will be missing so have to add in here
        sentences[z].append('[SEP]')
        tags[z].append('SEP')
        #pdb.set_trace()
        ee_ent_tag_ids[z].append((tag, tag_id))
        if len(tags[z]) != len(ee_ent_tag_ids[z]):
            # pdb.set_trace()
            raise Exception('nope')

        ent_sets[z] = entity_set
        ee_sentences, ee_labels, tag2wid,  ids2arg_sent, snippet_entity_pairs_captured = create_re_examples(tag2wid, ids2arg,
                                                     sentences,
                                                     trigger_set, entity_set,
                                                     previous_tagid, w, z,
                                                     ee_sentences, ee_labels,
                                                     ids2arg_sent, snippet_entity_pairs_captured)
        if len(ids2arg_sent[z]) != 2:
            pdb.set_trace()
            raise Exception('tuple size for ids and args are not equal to 2')

        tag2wids[z] = tag2wid
        #ee_ids2arg = [ids2arg] * len(sentences)


        # if len(sentences) != len(ee_ent_tag_ids):
        #     pdb.set_trace()
        #     raise Exception('Entity tag tuples do not match sentence length')
        # catch list of relations that are missed due to sentence tokenization
        # missed_entity_pairs = defaultdict(lambda: 0)
        # pdb.set_trace()

        for (trigger_id, entity_id), relation_label in ids2arg.items():
            if (trigger_id, entity_id) not in snippet_entity_pairs_captured:
                trigger_label = snippet_id_2_label[trigger_id]
                entity_label = snippet_id_2_label[entity_id]
                full_label = relation_label + '|' + trigger_label + '|' + entity_label
                missed_entity_pairs[full_label] += 1

    return sentences, tags, ee_sentences, ee_labels,  ids2arg_sent , ee_ent_tag_ids, ent_sets, tag2wids, missed_entity_pairs


def read_folder_ee(path):
    files = sorted(os.listdir(path))
    sents_all = []
    tags_all = []
    sents_ee_all = []
    labels_ee_all = []
    ee_ids2arg_all = []
    ee_ent_tag_ids_all = []
    ent_sets_all = []
    tag2wids_all = []
    missed_entity_pairs = defaultdict(def_value_missed_pairs)

    for s_file, e_file in zip(files[:-1], files[1:]):
        if s_file[-3:] == 'ann' and e_file[-3:] == 'txt':
            s_id = re.findall(r'\d+', s_file)
            e_id = re.findall(r'\d+', e_file)
            if s_id[0] == e_id[0]:
                sents, tags, ee_sents, ee_labels, ee_ids2arg, ee_ent_tag_ids, ent_sets, tag2wids, missed_entity_pairs = read_brat_format_ee(path + s_file, path + e_file, missed_entity_pairs)
                sents_all.extend(sents)
                tags_all.extend(tags)
                sents_ee_all.extend(ee_sents)
                labels_ee_all.extend(ee_labels)
                ee_ids2arg_all.extend(ee_ids2arg)
                ee_ent_tag_ids_all.extend(ee_ent_tag_ids)
                ent_sets_all.extend(ent_sets)
                tag2wids_all.extend(tag2wids)
                if len(sents) != len(ee_ent_tag_ids):
                    pdb.set_trace()
                    raise Exception('Entity tag tuples do not match sentence length')

    # if len(sents_all) != len(ee_ent_tag_ids_all):
    #     pdb.set_trace()
    #     raise Exception('Entity tag tuples do not match sentence length')

    print('missed pairs due to sentence tokenization summary')
    for k, v in missed_entity_pairs.items():
        print(f'type: {k}, total number: {v}')

    for i, (s, e_t_id) in enumerate(zip(sents_all, ee_ent_tag_ids_all)):
        if len(s) != len(e_t_id):
            pdb.set_trace()
            raise Exception(f'size mismatch at {i}')

    return sents_all, tags_all, sents_ee_all, labels_ee_all, ee_ids2arg_all, ee_ent_tag_ids_all, ent_sets_all, tag2wids_all, missed_entity_pairs


# def map2ind(tags):
#     #tag2i = {t: i+2 for i, t in enumerate(set(tag for sent in tags for tag in sent))}
#     tag2i = {}
#     tag2i['O'] = 0
#     tag2i['ARGM'] = 1
#     tag2i['ARG1'] = 2
#     i2tag = {i:t for t, i in tag2i.items()}
#
#     return tag2i, i2tag


class EvalDataset(Dataset):
    def __init__(self, sents, ids2arg, ent_tag_ids, ent_sets, tag2wids):
        self.sents = sents
        self.ids2arg = ids2arg
        self.ent_tag_ids = ent_tag_ids
        self.ent_sets = ent_sets
        self.tag2wids = tag2wids

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx], self.ids2arg[idx], self.ent_tag_ids[idx], self.ent_sets[idx], self.tag2wids[idx]


# def transformer_collate_fn(batch, tokenizer, label2id, use_wordpiece=False):
#     # pdb.set_trace()
#     bert_vocab = tokenizer.get_vocab()
#     bert_pad_token = bert_vocab['[PAD]']
#     trig_start_token = bert_vocab['[$]']
#     trig_end_token = bert_vocab['[/$]']
#     ent_start_token = bert_vocab['[#]']
#     ent_end_token = bert_vocab['[/#]']
#     unk_token = bert_vocab['[UNK]']
#
#     sentences, mask, labels, trig_mask, ent_mask = [], [], [], [], []
#
#     for data, target in batch:
#
#         tokenized_label = label2id[target]
#
#         if use_wordpiece:
#             tokenized = tokenizer(data, is_split_into_words=True,
#                                   return_offsets_mapping=True,
#                                   add_special_tokens=False)
#             tokenized_sent = tokenized.input_ids
#         else:
#             tokenized_sent = tokenizer.convert_tokens_to_ids(data)
#
#         if len(tokenized_sent) == 512:
#             raise Exception("maximum sentence length reached. sentence may have been truncated and could be missing"
#                             "relation tags")
#
#         mask.append(torch.ones(len(tokenized_sent)))
#         sentences.append(torch.tensor(tokenized_sent))
#         labels.append(torch.tensor(tokenized_label))
#         trig_mask.append((tokenized_sent.index(trig_start_token),
#                           tokenized_sent.index(trig_end_token)))
#         ent_mask.append((tokenized_sent.index(ent_start_token),
#                          tokenized_sent.index(ent_end_token)))
#
#     sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
#     mask = pad_sequence(mask, batch_first=True, padding_value=bert_pad_token)
#
#     return sentences, mask, torch.tensor(labels), trig_mask, ent_mask

def transformer_collate_fn(batch, tokenizer, use_wordpiece=False):

    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']

    sentences, mask, ids2args_all, ent_tag_ids_all, ent_sets_all, tag2wids_all = [], [], [], [], [], []

    for sent, ids2args, ent_tag_ids, ent_sets, tag2wid in batch:

        if use_wordpiece:
            tokenized = tokenizer(sent, is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  add_special_tokens=False)
            tokenized_sent = tokenized.input_ids
        else:
            tokenized_sent = tokenizer.convert_tokens_to_ids(sent)

        mask.append(torch.ones(len(tokenized_sent)))
        sentences.append(torch.tensor(tokenized_sent))
        ids2args_all.append(ids2args)
        ent_tag_ids_all.append(ent_tag_ids)
        ent_sets_all.append(ent_sets)
        tag2wids_all.append(tag2wid)

        if len(torch.tensor(tokenized_sent)) != len(ent_tag_ids):
            pdb.set_trace()
            raise Exception('Entity tag tuples do not match sentence length')

    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    mask = pad_sequence(mask, batch_first=True, padding_value=bert_pad_token)

    return sentences, mask, ids2args_all, ent_tag_ids_all, ent_sets_all, tag2wids_all


# def map2ind():
#     label2i = dict()
#     label2i['O'] = 0
#     label2i['ARGM'] = 1
#     label2i['ARG1'] = 2
#     i2label = {i: t for t, i in label2i.items()}
#     return label2i, i2label


def store_data():
    # sentences_train, tags_train, ee_sentences_train, ee_labels_train, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_all_train, tag2wids_all_train = read_folder_ee(
    #     'data/ee_train/')
    sentences_val, tags_val, ee_sentences_val, ee_labels_val, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_all_val, tag2wids_all_val, missed_entity_pairs_val = read_folder_ee(
        'data/ee_dev/')

    for s, e_t_id in zip(sentences_val, ee_ent_tag_ids_val):
        if len(s) != len(e_t_id):
            raise Exception('size mismatch')

        # train_sents_1, train_tags_1 = read_folder('data/train/')
    val_sents_1, val_tags_1 = preprocessor.read_folder('data/dev/')
    val_sents_2, val_tags_2 = preprocessor.read_folder('data/ee_dev/', labels=True, ner_task=2)

    tag2i, i2tag = preprocessor.map2ind(val_tags_1)
    trigger2i, i2trigger = preprocessor.map2ind(val_tags_2)

    #data_train = (sentences_train, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_all_train, tag2wids_all_train)
    data_val = (i2tag, i2trigger, sentences_val, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_all_val, tag2wids_all_val)

    with open("data/ee_eval_data.pickle", "wb") as f:
        pickle.dump(data_val, f)

    with open("data/missed_entity_pairs.pickle", "wb") as f:
        pickle.dump(missed_entity_pairs_val, f)


    return None


def colab_load_data(model_name, data, data_size=1):

    #sentences_train, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_train, tag2wids_train = data[0]
    i2tag, i2trigger, sentences_val, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_val, tag2wids_val = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

    #train_ind = int(data_size * len(sentences_train))
    val_ind = int(data_size * len(sentences_val))

    for s, e_t_id in zip(sentences_val, ee_ent_tag_ids_val):
        if len(s) != len(e_t_id):
            raise Exception('size mismatch')

    # train_dataset_re = EvalDataset(sentences_train[:train_ind],
    #                                ee_ids2arg_train[:train_ind],
    #                                ee_ent_tag_ids_train[:train_ind],
    #                                ent_sets_train[:train_ind],
    #                                tag2wids_train[:train_ind])

    val_dataset_re = EvalDataset(sentences_val[:val_ind],
                                 ee_ids2arg_val[:val_ind],
                                 ee_ent_tag_ids_val[:val_ind],
                                 ent_sets_val[:val_ind],
                                 tag2wids_val[:val_ind])

    biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add relation tokens
    biobert_tokenizer.add_special_tokens({'additional_special_tokens': ['[$]', '[/$]', '[#]', '[/#]']})


    # i2label = {i:t for t, i in label2i.items()}

    # train_dataloader = DataLoader(train_dataset_re, batch_size=32,
    #                               collate_fn=partial(transformer_collate_fn,
    #                                                  tokenizer=biobert_tokenizer,
    #                                                  use_wordpiece=False),
    #                               shuffle=True)

    val_dataloader = DataLoader(val_dataset_re, batch_size=32,
                                collate_fn=partial(transformer_collate_fn,
                                                   tokenizer=biobert_tokenizer,
                                                   use_wordpiece=False),
                                shuffle=False)

    # return sentences_train, tags_train, ee_sentences_train, ee_labels_train
    #return train_dataloader, val_dataloader, biobert_tokenizer
    return i2tag, i2trigger, val_dataloader, biobert_tokenizer


def measure_ee_f1(model_ner, model_re, dataloader, device, i2arg, i2trigger, i2tag, tokenizer):

    start_time = time.time()
    ''' this will only work with not using wordpiece '''
    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    trig_start_token = bert_vocab['[$]']
    trig_end_token = bert_vocab['[/$]']
    ent_start_token = bert_vocab['[#]']
    ent_end_token = bert_vocab['[/#]']
    unk_token = bert_vocab['[UNK]']

    i2trigger = {3: 'CLS', 0: 'WORKUP', 4: 'REACTION_STEP', 1: 'O', 2: 'SEP'}
    #trigger2i = {t: i for i, t in i2trigger.items() if i != 0 and i != 1}
    trigger2i = {t: i for i, t in i2trigger.items()}
    num_tags = len(trigger2i)

    model_ner.eval()
    model_re.eval()
    results_dict = {}
    for arg in i2arg.values():
        if arg == 'ARG1' or arg == 'ARGM':
            for trig in i2trigger.values():
                if trig == 'REACTION_STEP' or trig == 'WORKUP':
                    for tag in i2tag.values():
                        if tag != 'M' and tag != 'SPECIAL!' and tag != 'CLS' and tag != 'SEP' and tag != 'O':
                            results_dict[arg+'|'+trig+'|'+tag] = {'tp': 0, 'fp': 0, 'fn': 0}
    #results_dict = {arg+'|'+trig+'|'+tag: {'tp': 0, 'fp': 0, 'fn': 0} for arg in i2arg.values() for trig in i2trigger.values() for tag in i2tag.values() if trig != 'M' and trig != 'SPECIAL!' and trig != 'O' and arg != 'O' and tag != 'O'}
    print('\n beginning model pipeline evaluation...')
    with torch.no_grad():
        for sentences, mask, ids2args_all, ent_tag_ids_all, ent_sets_all, tag2wids_all in tqdm(dataloader):
            #print('\n beginning batch')
            outputs_ner = model_ner(sentences.to(device), attention_mask=mask.to(device))
            outputs_ner = outputs_ner.logits[:, :, :num_tags].argmax(dim=-1)

            for q, output_ner in enumerate(outputs_ner):
                #print('\nrunning RE model for one sentence')
                re_examples = []
                trig_mask = []
                ent_mask = []
                re_masks = []
                input_ents_re_ids = []
                entity_set = ent_sets_all[q]
                tag2wid = tag2wids_all[q]
                # ids2args = ids2args_all[q]
                # pdb.set_trace()
                ids2args, ids2arg_names = ids2args_all[q]
                ent_tag_ids = ent_tag_ids_all[q]
                padding_mask = mask[q]
                fixed_sent_labels = output_ner[padding_mask.bool()].detach().tolist()
                fixed_sent = sentences[q][padding_mask.bool()].detach().tolist()

                # # get rid of the CLS and SEP tokens for now
                # fixed_sent = fixed_sent[1:-1]
                # fixed_sent_labels = fixed_sent_labels[1:-1]


                trig_inds = []
                previous_tag = ''
                trig_id = 0
                # note this method will fail if two different trigger word tags are right next to each other
                ''' well it just so happens that that can happen even though it's not in the dataset back to the drawing board'''

                for i, tag in enumerate(fixed_sent_labels):
                    # assuming trigger entity is only one word
                    if tag == trigger2i['REACTION_STEP'] or tag == trigger2i['WORKUP']:
                        trig_inds.append({'start': i,
                                          'end': i+1,
                                          'trigger': i2trigger[tag],
                                          'gold_id': ent_tag_ids[i][1]})
                    # if tag != previous_tag:
                    #     if len(trig_inds) != 0 and previous_tag != trigger2i['O'] and previous_tag != trigger2i['M'] and previous_tag != trigger2i['SPECIAL!']:
                    #         trig_inds[trig_id]['end'] = i
                    #         trig_id += 1
                    #     if tag != trigger2i['O'] and tag != trigger2i['M'] and tag != trigger2i['SPECIAL!']:
                    #         trig_inds[trig_id] = {'start': i,
                    #                               'trigger': i2trigger[tag],
                    #                               'gold_id': ent_tag_ids[i][1]}
                    #     previous_tag = tag
                #pdb.set_trace()
                for t_id in range(0, len(trig_inds)):
                    # if a trigger word is predicted for a SEP or CLS token skip it
                    if trig_inds[t_id]['end'] == len(fixed_sent_labels) or trig_inds[t_id]['start'] == 0:
                        continue

                    trigger = trig_inds[t_id]['trigger']
                    for entity, entity_id in entity_set:

                        trig_start = trig_inds[t_id]['start']
                        trig_end = trig_inds[t_id]['end'] + 1 # add one to adjust for first insert of start

                        ent_start = tag2wid[entity_id + 'start']
                        # try:
                        ent_end = tag2wid[entity_id + 'end'] + 1  # add one to adjust for first insert of start

                        # if the entity span is equal to or contained within the tag span skip the example
                        if (ent_start >= trig_start and ent_end <= trig_end) or (trig_start >= ent_start and trig_end <= ent_end):
                            continue

                        adjust_sent = fixed_sent.copy()
                        input_ents_re_ids.append(((trigger, trig_inds[t_id]['gold_id']), (entity, entity_id)))
                        # trig_start = trig_inds[t_id]['start']
                        # trig_end = trig_inds[t_id]['end'] + 1 # add one to adjust for first insert of start

                        adjust_sent.insert(trig_start, trig_start_token)
                        adjust_sent.insert(trig_end, trig_end_token)

                        trig_num_pops = trig_end - trig_start - 1
                        del adjust_sent[trig_start + 1:trig_end]

                        if trig_start_token not in adjust_sent or trig_end_token not in adjust_sent:
                            raise('missing trigger token')

                        # try:
                        #     adjust_sent.insert(trig_start + 1, tokenizer.convert_tokens_to_ids(trigger))
                        # except:
                        #     pdb.set_trace()
                        adjust_sent.insert(trig_start + 1, tokenizer.convert_tokens_to_ids(trigger))

                        # ent_start = tag2wid[entity_id + 'start']
                        # #try:
                        # ent_end = tag2wid[entity_id + 'end'] + 1  # add one to adjust for first insert of start
                        #except:
                        #    pdb.set_trace()

                        # account for shift in the indices from the trigger inserts or deletes
                        if ent_start > trig_start:
                            ent_start += 2 + 1 - trig_num_pops
                            ent_end += 2 + 1 - trig_num_pops

                        adjust_sent.insert(ent_start, ent_start_token)
                        adjust_sent.insert(ent_end, ent_end_token)

                        del adjust_sent[ent_start + 1:ent_end]
                        adjust_sent.insert(ent_start + 1, tokenizer.convert_tokens_to_ids(entity))

                        # if ent_start_token not in adjust_sent or ent_start_token not in adjust_sent:
                        #     raise Exception('missing trigger token')
                        #     pdb.set_trace()
                        #
                        # if trig_start_token not in adjust_sent or trig_end_token not in adjust_sent:
                        #     raise Exception('missing trigger token')
                        #     pdb.set_trace()


                        try:
                            adjust_sent.index(trig_start_token)
                        except:
                            pdb.set_trace()

                        try:
                            adjust_sent.index(trig_end_token)
                        except:
                            pdb.set_trace()

                        try:
                            adjust_sent.index(ent_start_token)
                        except:
                            pdb.set_trace()
                        try:
                            adjust_sent.index(ent_end_token)
                        except:
                            pdb.set_trace()




                        #try:
                        # add CLS and SEP tokens back
                        # adjust_sent.insert(0, '[CLS]')
                        # adjust_sent.insert(-1, '[SEP]')
                        re_examples.append(torch.tensor(adjust_sent))
                        trig_mask.append((adjust_sent.index(trig_start_token),
                                          adjust_sent.index(trig_end_token)))
                        ent_mask.append((adjust_sent.index(ent_start_token),
                                         adjust_sent.index(ent_end_token)))
                        re_masks.append(torch.ones(len(adjust_sent)))
                        #except:
                            #pdb.set_trace()
                # pdb.set_trace()
                gold_label_keys_unused = list(ids2arg_names.keys())
                if len(re_examples) > 0 and len(re_masks) > 0:
                    # in the future need to catch other cases
                    # dont know if it is faster to use batches or not
                    try:
                        re_sentences = pad_sequence(re_examples, batch_first=True, padding_value=bert_pad_token).to(device)
                    except:
                        pdb.set_trace()
                    re_masks = pad_sequence(re_masks, batch_first=True, padding_value=bert_pad_token).to(device)
                    outputs_re = model_re(re_sentences, re_masks, trig_mask, ent_mask)
                    outputs_re = torch.argmax(outputs_re, dim=1)

                    if ids2args.keys() != ids2arg_names.keys():
                        raise Exception('i2arg tuples are not the same length')

                    for arg_label, ((trigger, trigger_id), (entity, entity_id)) in zip(outputs_re, input_ents_re_ids):
                        if (trigger_id, entity_id) in ids2args:
                            gold_label = ids2args[(trigger_id, entity_id)]
                            if (trigger_id, entity_id) in gold_label_keys_unused:
                                gold_label_keys_unused.remove((trigger_id, entity_id))
                        else:
                            gold_label = 'O'
                        # try:
                        pred = i2arg[arg_label.item()]
                        # except:
                            # pdb.set_trace()
                        actual = gold_label
                        full_pred_name = pred+'|'+trigger+'|'+entity
                        full_actual_name = actual+'|'+trigger+'|'+entity

                        pred_arg, pred_trig, pred_ent = full_pred_name.split('|')
                        actual_arg, actual_trig, actual_ent = full_actual_name.split('|')

                        pred_status = sum([e == 'O' or e == 'M' or e == 'SEP' or e == 'CLS' or e == 'SPECIAL!' for e in full_pred_name.split('|')])
                        actual_status = sum([e == 'O' or e == 'M' or e == 'SEP' or e == 'CLS' or e == 'SPECIAL!' for e in full_actual_name.split('|')])

                        if pred == actual and pred_status == 0:
                            results_dict[full_actual_name]['tp'] += 1
                        elif pred_status == 0 or actual_status == 0:
                            if pred_status != 0 and actual_status == 0:
                                results_dict[full_actual_name]['fn'] += 1
                            elif actual_status != 0 and pred_status == 0:
                                results_dict[full_pred_name]['fp'] += 1
                            else:
                                results_dict[full_actual_name]['fn'] += 1
                                results_dict[full_pred_name]['fp'] += 1
                        else:
                            pass

                # catch false negatives from labels not developed in inference but in gold sentence
                if len(gold_label_keys_unused) > 0:
                    for (trigger_id, entity_id) in gold_label_keys_unused:
                        label = ids2args[(trigger_id, entity_id)]
                        if label != 'O':
                            trigger, entity = ids2arg_names[(trigger_id, entity_id)]
                            results_dict[label+'|'+trigger+'|'+entity]['fn'] += 1
                #print('\ncompleted RE model section for one sentence of batch')
            #print('\nfinished batch')



    f1_scores = []

    gold_label_result_keys = ['ARG1|REACTION_STEP|OTHER_COMPOUND',
                              'ARG1|REACTION_STEP|REACTION_PRODUCT',
                              'ARG1|REACTION_STEP|REAGENT_CATALYST',
                              'ARG1|REACTION_STEP|SOLVENT',
                              'ARG1|REACTION_STEP|STARTING_MATERIAL',
                              'ARG1|WORKUP|OTHER_COMPOUND',
                              'ARG1|WORKUP|REACTION_PRODUCT',
                              'ARG1|WORKUP|SOLVENT',
                              'ARG1|WORKUP|STARTING_MATERIAL',
                              'ARGM|REACTION_STEP|TEMPERATURE',
                              'ARGM|REACTION_STEP|TIME',
                              'ARGM|REACTION_STEP|YIELD_OTHER',
                              'ARGM|REACTION_STEP|YIELD_PERCENT',
                              'ARGM|WORKUP|TEMPERATURE',
                              'ARGM|WORKUP|TIME',
                              'ARGM|WORKUP|YIELD_OTHER',
                              'ARGM|WORKUP|YIELD_PERCENT']

    # catch the cases for false negatives of missed pairs due to sentence segmentation
    with open("data/missed_entity_pairs.pickle", "rb") as f:
        missed_entity_pairs = pickle.load(f)

    for label_type, num_missed in missed_entity_pairs.items():
        results_dict[label_type]['fn'] += num_missed

    tp_total = 0
    fn_total = 0
    fp_total = 0
    print("\\begin{table}[t] \n \\caption{results of base custom relation extraction model} \n \\label{tab:table_name}\n \\begin{tabular}{||cccc||}\n \\hline")
    print("Relation Type & Precision & Recall & F1    \\\\ \n \\hline \\hline")
    for label in results_dict.keys():
        if label in gold_label_result_keys:
        # if sum([e == 'O' for e in label.split('|')]) == 0:
            precision = round(
                results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fp'] + 1e-6),
                4)
            recall = round(
                results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fn'] + 1e-6),
                4)
            f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)
            tp_total += results_dict[label]['tp']
            fn_total += results_dict[label]['fn']
            fp_total += results_dict[label]['fp']
            if f1 > 0.0000:
                f1_scores.append(f1)
            print(f"{label} & {precision :.3f} & {recall :.3f} & {f1 :.3f}    \\\\ \n \\hline")
            #print("Event Label: ", label, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

    precision_overall = round(tp_total / (tp_total + fp_total + 1e-6), 4)
    recall_overall = round(tp_total / (tp_total + fn_total + 1e-6), 4)
    f1_overall = round((2 * precision_overall * recall_overall) / (precision_overall + recall_overall + 1e-6), 4)
    f1_average = sum(f1_scores) / len(f1_scores)
    #print('Average non-zero F1 Score: ', round(f1_average, 4))

    print(f"\\hline \n overall & {precision_overall :.3f} & {recall_overall :.3f} & {f1_overall :.3f}    \\\\ \n \\hline")
    print('\\end{tabular} \n \\end{table}')

    print("Overall Performance, Precision: ", precision_overall, ", Recall: ", recall_overall, ", F1: ", f1_overall)
    end_time = time.time()
    #print(f'evaluation took:{(end_time - start_time)/60 :.2f} minutes')
    return f1_average



                #output = torch.cat([output[:trig_inds[t_id]['start']+z*2], trig_start_token, output[trig_inds[t_id]['start']+z*2:]], 0)
                #output = torch.cat([output[:trig_inds[t_id]['end']+z*2+1], trig_end_token, output[trig_inds[t_id]['end']+z*2+1:]], 0)









