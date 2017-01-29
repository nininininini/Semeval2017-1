from gensim.models import word2vec
from ufal_udpipe import *
from nltk import word_tokenize
from sys import stdout as out
import numpy as np
import regex as re


# loads pretrained embeddings from Word2Vec (Gensim module)
def load_gensim_w2v(filename):
    try:
        embedding_model = word2vec.Word2Vec.load(filename)
        # extract vocabulary for indexing
        vocab = [w for (w, _) in embedding_model.vocab.items()]
        # extract weights for keras layer
        embedding_weights = [np.array([embedding_model[w] for w in vocab])]
        return vocab, embedding_weights
    except:
        out.write('Gensim Word2Vec file is missing\n')


# return the sentence analysis
class UDpipe_analyser:
    def __init__(self):
        self.model = Model.load('english-ud-1.2-160523.udpipe')
        self.pipeline = Pipeline(self.model, 'horizontal', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    def udpipe_analysis(self, sent):
        analysis = self.pipeline.process(' '.join(word_tokenize(sent)))
        # remove punctuation
        tags = [t.split()[3:6] for t in analysis.split('\n') if t and t.split()[3] != 'PUNCT']
        tags = [t[:2] + re.findall(r'(?<=\=)[^|]+(?=(?:\||$))', t[-1]) for t in tags]
        # print only tags
        return ' '.join([''.join(t) for t in tags])
