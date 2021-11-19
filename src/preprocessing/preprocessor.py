from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import re


def read_brat_format(txt_filename, ann_filename):

    # read in raw text from patent snippet
    snippet = ""
    for line in open(txt_filename).readlines():
        snippet += line
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
            sentences[z].append('-END-')
            tags[z].append('END')
            z += 1

        # if the current sentence is empty place the START token
        if len(sentences[z]) == 0:
            sentences[z] = ['-START-']
            tags[z] = ['START']

        sentences[z].append(word)
        tags[z].append(tag)

    # last iterations ending tokens will be missing so have to add in here
    sentences[z].append('-END-')
    tags[z].append('END')

    return sentences, tags


def load_data():

    train_sents, train_tags = read_brat_format('data/train/0000.txt', 'data/train/0000.ann')
    print(train_sents)
    print(train_tags)

    return train_sents, train_tags