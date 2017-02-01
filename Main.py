#!/usr/bin/env python3
from Model import TGVModel
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg import norm
import numpy as np
import pickle
import sys

data_file = sys.argv[1]
batch_s = int(sys.argv[2])
combine = int(sys.argv[3])
word_layer = int(sys.argv[4])
testing_data = sys.argv[5]
output_data = sys.argv[6]

with open(data_file, "rb") as f:
    word_embedding_matrix, tag_embedding_matrix, trigrams_e, word_e, tags_e, golden = pickle.load(f)

# pad the sequences for more efficient processing
tags_e = pad_sequences(tags_e)
word_e = pad_sequences(word_e)
trigrams_e = pad_sequences(trigrams_e)
gol = [[e] for e in golden]
desired = np.array(gol)

# create a new model
m = TGVModel(word_embedding_matrix, tag_embedding_matrix, word=word_layer, combining=combine)

# first 1530 are the trains
# 1530: are the training data
epochs = 0
for e in [18, 1, 1, 1, 1]:
    # this should be improved
    m.model.fit([trigrams_e[:1530], word_e[:1530], tags_e[:1530]], desired[:1530], batch_size=batch_s, nb_epoch=e)
    epochs += e
    # testing data
    golden_values = [x * 2 - 1 for x in golden[1530:]]
    # use cross-validation later
    prediction = m.predict([trigrams_e[1530:], word_e[1530:], tags_e[1530:]]) * 2 - 1
    prediction = prediction * 2 - 1
    sys.stdout.write(
        'INFO: ' + str(epochs) + ' epochs: ' + str(norm(prediction) / norm(golden_values)) + ' * ' + str(1 -
                                                                                                         cosine(
                                                                                                             prediction,
                                                                                                             golden_values)) + '\n')

with open(testing_data, "rb") as f:
    ids, cash_tags, trigrams_test, word_test, tags_test = pickle.load(f)

trigrams_test = pad_sequences(trigrams_test)
word_test = pad_sequences(word_test)
tags_test = pad_sequences(tags_test)

# final evaluation
prediction = m.predict([trigrams_test, word_test, tags_test])
prediction = prediction * 2 - 1

f2 = open('test2.json', 'w')
output = []
import json

for i in range(len(ids)):
    data = {'id': ids[i],'cashtag': cash_tags[i],'sentiment score': prediction[i]}
    output.append(data)

json.dump(output, f2, indent=4, sort_keys=True, separators=(',', ':'))
