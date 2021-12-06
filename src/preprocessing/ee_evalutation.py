

from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

import pdb

from functools import partial
from transformers import AutoTokenizer, AutoModel

import torch

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

    return ee_sentences, ee_labels


def read_brat_format_ee(ann_filename, txt_filename):
    # read in raw text from patent snippet
    snippet = ""
    for line in open(txt_filename, encoding="utf8").readlines():
        snippet += line

    if ann_filename is not None:
        # pdb.set_trace()

        # create an array of tags with O
        char_tags = ['O'] * len(snippet)
        id_tags = ['O'] * len(snippet)
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
                ent_sets[z] = entity_set
                tag2wids[z] = tag2wid
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

            # if the current sentence is empty place the START token
            if len(sentences[z]) == 0:
                sentences[z] = ['[CLS]']
                tags[z] = ['CLS']
                ee_ent_tag_ids[z].append(('O', 'O'))
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
            ee_ent_tag_ids[z].append((tag, tag_id))

            w += 1

        # last iterations ending tokens will be missing so have to add in here
        sentences[z].append('[SEP]')
        tags[z].append('SEP')
        ee_ent_tag_ids[z].append((tag, tag_id))
        ent_sets[z] = entity_set
        tag2wids[z] = tag2wid
        ee_sentences, ee_labels = create_re_examples(tag2wid, ids2arg,
                                                     sentences,
                                                     trigger_set, entity_set,
                                                     previous_tagid, w, z,
                                                     ee_sentences, ee_labels)
        ee_ids2arg = [ids2arg] * len(sentences)

    return sentences, tags, ee_sentences, ee_labels, ee_ids2arg, ee_ent_tag_ids, ent_sets, tag2wids


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

    for s_file, e_file in zip(files[:-1], files[1:]):
        if s_file[-3:] == 'ann' and e_file[-3:] == 'txt':
            s_id = re.findall(r'\d+', s_file)
            e_id = re.findall(r'\d+', e_file)
            if s_id[0] == e_id[0]:
                sents, tags, ee_sents, ee_labels, ee_ids2arg, ee_ent_tag_ids, ent_sets, tag2wids = read_brat_format_ee(path + s_file, path + e_file)
                sents_all.extend(sents)
                tags_all.extend(tags)
                sents_ee_all.extend(ee_sents)
                labels_ee_all.extend(ee_labels)
                ee_ids2arg_all.extend(ee_ids2arg)
                ee_ent_tag_ids_all.extend(ee_ent_tag_ids)
                ent_sets_all.extend(ent_sets)
                tag2wids_all.extend(tag2wids)


    return sents_all, tags_all, sents_ee_all, labels_ee_all, ee_ids2arg_all, ee_ent_tag_ids_all, ent_sets_all, tag2wids_all


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

    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    mask = pad_sequence(mask, batch_first=True, padding_value=bert_pad_token)

    return sentences, mask, ids2args_all, ent_tag_ids_all, ent_sets_all, tag2wids_all


def map2ind():
    label2i = dict()
    label2i['O'] = 0
    label2i['ARGM'] = 1
    label2i['ARG1'] = 2
    i2label = {i: t for t, i in label2i.items()}
    return label2i, i2label


def store_data():
    sentences_train, _, _, _, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_all_train, tag2wids_all_train = read_folder_ee(
        'data/ee_train/')
    sentences_val, _, _, _, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_all_val, tag2wids_all_val = read_folder_ee(
        'data/ee_dev/')

    data_train = (sentences_train, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_all_train, tag2wids_all_train)
    data_val = (sentences_val, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_all_val, tag2wids_all_val)

    with open("data/ee_eval_data.pickle", "wb") as f:
        pickle.dump((data_train, data_val), f)

    return None


def colab_load_data(model_name, data, data_size=1):

    sentences_train, ee_ids2arg_train, ee_ent_tag_ids_train, ent_sets_train, tag2wids_train = data[0]
    sentences_val, ee_ids2arg_val, ee_ent_tag_ids_val, ent_sets_val, tag2wids_val = data[1]

    train_ind = int(data_size * len(sentences_train))
    val_ind = int(data_size * len(sentences_val))

    train_dataset_re = EvalDataset(sentences_train[:train_ind],
                                   ee_ids2arg_train[:train_ind],
                                   ee_ent_tag_ids_train[:train_ind],
                                   ent_sets_train[:train_ind],
                                   tag2wids_train[:train_ind])

    val_dataset_re = EvalDataset(sentences_val[:val_ind],
                                 ee_ids2arg_val[:val_ind],
                                 ee_ent_tag_ids_val[:val_ind],
                                 ent_sets_val[:val_ind],
                                 tag2wids_val[:val_ind])

    biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add relation tokens
    biobert_tokenizer.add_special_tokens({'additional_special_tokens': ['[$]', '[/$]', '[#]', '[/#]']})

    label2i, _ = map2ind()
    # i2label = {i:t for t, i in label2i.items()}

    train_dataloader = DataLoader(train_dataset_re, batch_size=32,
                                  collate_fn=partial(transformer_collate_fn,
                                                     tokenizer=biobert_tokenizer,
                                                     use_wordpiece=False),
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset_re, batch_size=32,
                                collate_fn=partial(transformer_collate_fn,
                                                   tokenizer=biobert_tokenizer,
                                                   use_wordpiece=False),
                                shuffle=True)

    # return sentences_train, tags_train, ee_sentences_train, ee_labels_train
    return train_dataloader, val_dataloader, biobert_tokenizer


def measure_ee_f1(model_ner, model_re, dataloader, device, i2arg, i2trigger, i2tag, tokenizer):

    ''' this will only work with not using wordpiece '''
    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    trig_start_token = bert_vocab['[$]']
    trig_end_token = bert_vocab['[/$]']
    ent_start_token = bert_vocab['[#]']
    ent_end_token = bert_vocab['[/#]']
    unk_token = bert_vocab['[UNK]']

    model_ner.eval()
    model_re.eval()
    results_dict = {arg+'|'+trig+'|'+tag for arg in i2arg.values() for trig in i2trigger.values() for tag in i2tag.values()}

    with torch.no_grad():
        for sentences, mask, ids2args_all, ent_tag_ids_all, ent_sets_all, tag2wids_all in dataloader:
            outputs_ner = model_ner(sentences.to(device), attention_mask=mask.to(device))

            for q, output_ner in enumerate(outputs_ner):
                re_examples = []
                trig_mask = []
                ent_mask = []
                masks = []
                input_ents_re_ids = []
                entity_set = ent_sets_all[q]
                tag2wid = tag2wids_all[q]
                ids2args = ids2args_all[q]
                ent_tag_ids = ent_tag_ids_all[q]
                fixed_sent = output_ner.item().tolist()

                trig_inds = {}
                previous_tag = ''
                trig_id = 0
                # note this method will fail if two different trigger word tags are right next to each other
                for i, tag in enumerate(output_ner):
                    if tag != previous_tag:
                        if len(trig_inds) != 0 and previous_tag != 'O':
                            trig_inds[trig_id]['end'] = i
                            trig_id += 1
                        if tag != 'O':
                            trig_inds[trig_id] = {'start': i, 'trigger': tag, 'gold_id':ent_tag_ids[i][0]}
                        previous_tag = tag

                for t_id in trig_inds.keys():
                    for entity, entity_id in entity_set:
                        adjust_sent = fixed_sent.copy()
                        input_ents_re_ids[((trigger, trig_inds[t_id]['gold_id']), (entity, entity_id))]
                        trig_start = trig_inds[t_id]['start']
                        trig_end = trig_inds[t_id]['end'] + 1 # add one to adjust for first insert of start
                        trigger = trig_inds[t_id]['tag']

                        adjust_sent.insert(trig_start, trig_start_token)
                        adjust_sent.insert(trig_end, trig_end_token)

                        trig_num_pops = trig_end - trig_start - 1
                        del adjust_sent[trig_start + 1:trig_end]
                        adjust_sent.insert(trig_start + 1, tokenizer.convert_tokens_to_ids(trigger))

                        ent_start = tag2wid[entity_id + 'start']
                        ent_end = tag2wid[entity_id + 'end'] + 1  # add one to adjust for first insert of start

                    # account for shift in the indices from the trigger inserts or deletes
                    if ent_start > trig_start:
                        ent_start += 2 + 1 - trig_num_pops
                        ent_end += 2 + 1 - trig_num_pops

                    adjust_sent.insert(ent_start, ent_start_token)
                    adjust_sent.insert(ent_end, ent_end_token)

                    del adjust_sent[ent_start + 1:ent_end]
                    adjust_sent.insert(ent_start + 1, tokenizer.convert_tokens_to_ids(entity))
                    re_examples.append(torch.tensor(adjust_sent))
                    trig_mask.append((adjust_sent.index(trig_start_token),
                                      adjust_sent.index(trig_end_token)))
                    ent_mask.append((adjust_sent.index(ent_start_token),
                                     adjust_sent.index(ent_end_token)))
                    masks.append(torch.ones(len(adjust_sent)))

                # dont know if it is faster to use batches or not
                sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
                masks = pad_sequence(masks, batch_first=True, padding_value=bert_pad_token)
                outputs_re = model_re(torch.stack(re_examples).to(device), masks.to(device), trig_mask, ent_mask)

                for arg_label, ((trigger, trigger_id), (entity, entity_id)) in zip(outputs_re, input_ents_re_ids):
                    gold_label = ids2args[(trigger_id, entity_id)]

                    pred = i2arg[arg_label.item()]
                    actual = gold_label
                    full_pred_name = pred+'|'+trigger+'|'+entity
                    full_actual_name = actual+'|'+trigger+'|'+entity

                    pred_arg, pred_trig, pred_ent = full_pred_name.split('|')
                    actual_arg, actual_trig, actual_ent = full_actual_name.split('|')

                    pred_status = sum([e == 'O' for e in full_pred_name.split('|')])
                    actual_status = sum([e == 'O' for e in  full_actual_name.split('|')])

                    if pred == actual and pred_status == 0:
                        results_dict[full_actual_name]['tp'] += 1
                    elif not (pred_status != 0 and actual_status != 0):
                        if (pred_arg == 'O' or pred_trig == 'O') and actual_status == 0:
                            results_dict[full_actual_name]['fn'] += 1
                        elif (actual_arg == 'O' or actual_trig == 'O') and pred_status == 0:
                            results_dict[full_pred_name]['fp'] += 1
                        else:
                            results_dict[full_actual_name]['fn'] += 1
                            results_dict[full_pred_name]['fp'] += 1
                    else:
                        pass

    f1_scores = []

    for label in results_dict.keys():
        if sum([e == 'O' for e in label.split('|')]) == 0:
            precision = round(
                results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fp'] + 1e-6),
                4)
            recall = round(
                results_dict[label]['tp'] / (results_dict[label]['tp'] + results_dict[label]['fn'] + 1e-6),
                4)
            f1 = round((2 * precision * recall) / (precision + recall + 1e-6), 4)
            f1_scores.append(f1)
            print("Event Label: ", label, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

    f1_average = sum(f1_scores) / len(f1_scores)
    print('Average F1 Score: ', round(f1_average, 4))
    return f1_average



                #output = torch.cat([output[:trig_inds[t_id]['start']+z*2], trig_start_token, output[trig_inds[t_id]['start']+z*2:]], 0)
                #output = torch.cat([output[:trig_inds[t_id]['end']+z*2+1], trig_end_token, output[trig_inds[t_id]['end']+z*2+1:]], 0)









