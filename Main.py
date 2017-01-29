#!/usr/bin/env python3
from Model import TGVModel
from Functions import UDpipe_analyser
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import regex as re
import numpy as np
import json
import sys

word_embeddings = sys.argv[1]
tags_embeddings = sys.argv[2]
training_data = sys.argv[3]
# training_data = '/home/vladan/semeval2017/semeval-2017-task-5-subtask-1/Microblog_Trainingdata.json'
model = TGVModel(sys.argv[1], sys.argv[2])
# dictionaries are faster for index lookup
vocabulary_dictionary = {k: v for v, k in enumerate(model.word_vocab)}
tag_dictionary = {k: v for v, k in enumerate(model.tags_vocab)}
A = UDpipe_analyser()
# all characters present in training data
# wasteful because most of the trigrams are not used
# Ain't nobody got memory for that
characters = ['T', 'G', 'W', 'o', 'a', '$', 'l', '7', 'S', '/', '3', 'x', 'r', 'Y', '4', 'X', 'F', '&', 'A', 'C', 'v',
              'd', '#', '8', 'L', 'h', '.', 'I', 'y', '@', 'N', '€', '6', 'K', '~', 'E', '1', '!', 'z', 'D', 'k', 'U',
              'V', 'm', 'e', 'f', 'w', 'B', '(', 'n', '+', 'j', '5', '=', '2', '<', '?', '>', 'R', '0', 'O', '-', ' ',
              'â', 'H', 'u', 'q', 's', 'Z', 'p', 't', 'J', "'", 'P', 'Q', 'b', '*', '%', ':', '9', 'c', ')', 'g', 'i',
              '™', 'M', ',']
all_trigrams = [a + b + c for a in characters for b in characters for c in characters]
# append OTHER trigram category
all_trigrams.append('OTH')
trigram_dictionary = {k: v for v, k in enumerate(all_trigrams)}

# preprocess data
with open(training_data) as data:
    data_train = json.load(data)
    golden = [x['sentiment score'] for x in data_train]
    golden = [(float(x) + 1) / 2 for x in golden]
    spans = [x['spans'] for x in data_train]
    # analyse everything
    tags_spans = [[A.udpipe_analysis(s).split() for s in span] for span in spans]
    # concatenate the sublists in training examples
    tags_spans = [[t for sublist in lists for t in sublist] for lists in tags_spans]
    # extract tags indices in the vocabulary
    tags_e = [[tag_dictionary[t] for t in example] for example in tags_spans]
    # preprocess sentences
    word_spans = [re.sub(r'[[:punct:]]', ' ', re.sub(r'\$[A-Z]*', 'company', ' '.join(example))).lower().split() for
                  example in spans]
    # extract words indices in the vocabulary
    word_e = [[vocabulary_dictionary[t] for t in example if t in model.word_vocab] for example in word_spans]
    # get trigrams in the sentences
    grams = [[[sentence[i:i + 3] for i in range(len(sentence) - 3 + 1)] for sentence in span] for span in spans]
    trigrams_e = [[trigram_dictionary[t] if t in trigram_dictionary else trigram_dictionary['OTH']
                   for e in example for t in e] for example in grams]

# first 1530 are the trains
# 1530: are the training data
for e in range(100):
    for p in range(1530):
        # this should be improved but Theano gave me chronic depression
        model.model.fit([np.array([trigrams_e[p]]), np.array([word_e[p]]), np.array([tags_e[p]])],
                        np.array([golden[p]]), nb_epoch=1)

    # testing data
    golden_values = [x * 2 - 1 for x in golden[1530:]]
    prediction = []
    # use cross-validation later
    for t in range(1530, 1700):
        prediction.append(
            model.predict([np.array([trigrams_e[t]]), np.array([word_e[t]]), np.array([tags_e[t]])])[0][0] * 2 - 1)
    sys.stdout.write(
        'INFO:' + e + 'epochs:' + norm(prediction) / norm(golden_values) * cosine(prediction, golden_values) + '\n')
