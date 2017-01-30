from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Merge, Dense, Dropout
import numpy as np


# Description of the used model
class TGVModel_tw:
    # word_network/vocab
    # tags_network/vocab
    def __init__(self, word_embedding_matrix, tags_embedding_matrix, trigram=25, word=64, tags=10, combining=32):
        #self.trigram_model = self.random_first_level_network(lstm_output_size=trigram)
        self.word_network = self.first_level_network(word_embedding_matrix, lstm_output_size=word)
        #self.tags_network = self.first_level_network(tags_embedding_matrix, lstm_output_size=tags)
        self.model = self.second_level_network(self.word_network, combining_layer=combining)

    # define first level model
    def first_level_network(self, embedding_matrix, lstm_output_size=64, combining_layer=32):
        # load embeddings from the model
        # get vocabulary and weights
        model = Sequential()
        # embedding (input) layer
        embedding_layer = Embedding(len(embedding_matrix[0]),  # vocabulary size \
                                    len(embedding_matrix[0][0]),  # output - embedding dimension  \
                                    weights=embedding_matrix,  # weight martix of embeddings \
                                    trainable=True)
        model.add(embedding_layer)
        # memory layer
        model.add(Dropout(0.1))
        model.add(LSTM(lstm_output_size))
        model.add(Dropout(0.1))

        # add a layer before outputing yet
        model.add(Dense(combining_layer, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        # final evaluation is based on the cosine similarity
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # wrapper functions
    def predict(self, x):
        return self.model.predict(x)
