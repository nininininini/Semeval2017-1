from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Merge, Dense
from Functions import load_gensim_w2v
import numpy as np


# Description of the used model
class TGVModel:
    # word_network/vocab
    # tags_network/vocab
    def __init__(self, word_embeddings, tags_embeddings):
        self.trigram_model = self.random_first_level_network()
        self.word_vocab, self.word_network = self.first_level_network(word_embeddings)
        self.tags_vocab, self.tags_network = self.first_level_network(tags_embeddings)
        self.model = self.second_level_network(self.trigram_model, self.word_network, self.tags_network)

    # define first level model
    def first_level_network(self, filename, lstm_output_size=64):
        # load embeddings from the model
        # get vocabulary and weights
        vocabulary, embedding_matrix = load_gensim_w2v(filename)
        model = Sequential()
        # embedding (input) layer
        embedding_layer = Embedding(len(vocabulary),  # vocabulary size \
                                    len(embedding_matrix[0][0]),  # output - embedding dimension  \
                                    weights=embedding_matrix,  # weight martix of embeddings \
                                    trainable=True)
        model.add(embedding_layer)
        # memory layer
        model.add(LSTM(lstm_output_size))
        return vocabulary, model

    # define random embedded layer
    def random_first_level_network(self, embedding_dimension=50, length=658504, lstm_output_size=64):
        model = Sequential()
        # embedding (input) layer
        embedding_layer = Embedding(length,  # number of trigrams \
                                    embedding_dimension,  # output - embedding dimension  \
                                    weights=[np.random.random_sample((length, embedding_dimension))],
                                    # weight martix of embeddings \
                                    trainable=True)
        model.add(embedding_layer)
        # memory layer
        model.add(LSTM(lstm_output_size))
        return model

    # concatenate the lower levels
    def second_level_network(self, grams, words, tags):
        # concatenate first layer models
        cat_model = Sequential()
        cat_model.add(Merge([grams, words, tags], mode='concat', concat_axis=1))
        # produce the final model
        final_model = Sequential()
        final_model.add(cat_model)
        # add a layer before outputing yet
        final_model.add(Dense(35, activation='sigmoid'))
        final_model.add(Dense(1, activation='sigmoid'))
        # final evaluation is based on the cosine similarity
        final_model.compile(loss='mean_squared_error', optimizer='adam')
        return final_model

    # wrapper functions
    def predict(self, x):
        return self.model.predict(x)
