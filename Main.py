#!/usr/bin/env python3
from Model import TGVModel
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg import norm
import numpy as np
import pickle
import sys

data_file = sys.argv[1]
batch_s = sys.argv[2]

with open(data_file, "rb") as f:
    word_embedding_matrix, tag_embedding_matrix, trigrams_e, word_e, tags_e, golden = pickle.load(f)

# pad the sequences for more efficient processing
tags_e = pad_sequences(tags_e)
word_e = pad_sequences(word_e)
trigrams_e = pad_sequences(trigrams_e)
gol = [[e] for e in golden]
desired = np.array(gol)

# create a new model
m = TGVModel(word_embedding_matrix, tag_embedding_matrix)

# first 1530 are the trains
# 1530: are the training data
epochs = 0
for e in [1, 1, 3, 5, 10, 15, 15, 25, 25, 25, 25, 25, 25]:
    # this should be improved
    m.model.fit([trigrams_e[:1530], word_e[:1530], tags_e[:1530]], desired[:1530], batch_size=batch_s, nb_epoch=e)
    epochs += e
    # testing data
    golden_values = [x * 2 - 1 for x in golden[1530:]]
    # use cross-validation later
    prediction = m.predict([trigrams_e[:1530], word_e[:1530], tags_e[:1530]]) * 2 - 1
    prediction = prediction * 2 -1
    sys.stdout.write(
        'INFO: ' + str(epochs) + ' epochs: ' + str(
            norm(prediction) / norm(golden_values) * cosine(prediction, golden_values)) + '\n')
