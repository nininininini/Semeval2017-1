#!/usr/bin/env python3
from Model import TGVModel
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import numpy as np
import pickle
import sys

data_file = sys.argv[1]

with open(data_file, "rb") as f:
    word_embedding_matrix, tag_embedding_matrix, trigrams_e, word_e, tags_e, golden = pickle.load(f)

# create a new model
m = TGVModel(word_embedding_matrix, tag_embedding_matrix)

# first 1530 are the trains
# 1530: are the training data
for e in range(50):
    for p in range(1530):
        # this should be improved but Theano gave me chronic depression
        m.model.fit([np.array([trigrams_e[p]]), np.array([word_e[p]]), np.array([tags_e[p]])],
                        np.array([golden[p]]), nb_epoch=1)

    # testing data
    golden_values = [x * 2 - 1 for x in golden[1530:]]
    prediction = []
    # use cross-validation later
    for t in range(1530, 1700):
        prediction.append(
            m.predict([np.array([trigrams_e[t]]), np.array([word_e[t]]), np.array([tags_e[t]])])[0][0] * 2 - 1)
    sys.stdout.write(
        'INFO:' + e + 'epochs:' + norm(prediction) / norm(golden_values) * cosine(prediction, golden_values) + '\n')
