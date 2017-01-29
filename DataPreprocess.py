from Functions import UDpipe_analyser
from Functions import load_gensim_w2v
import regex as re
import pickle
import sys
import json

word_embeddings = sys.argv[1]
tag_embeddings = sys.argv[2]
training_data = sys.argv[3]

# load word embeddings matrix
word_vocabulary, word_embedding_matrix = load_gensim_w2v(word_embeddings)
# dictionaries are faster for index lookup
word_dictionary = {k: v for v, k in enumerate(word_vocabulary)}

# load tags embedding matrix
tag_vocabulary, tag_embedding_matrix = load_gensim_w2v(tag_embeddings)
tag_dictionary = {k: v for v, k in enumerate(tag_vocabulary)}

# new UD Pipe analyser
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
# create a index-dictionary of trigrams
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
    tags_e = [[tag_dictionary[t] for t in example if t in tag_dictionary] for example in tags_spans]
    # preprocess sentences
    word_spans = [re.sub(r'[[:punct:]]', ' ', re.sub(r'\$[A-Z]*', 'company', ' '.join(example))).lower().split() for
                  example in spans]
    # extract words indices in the vocabulary
    word_e = [[word_dictionary[t] for t in example if t in word_dictionary] for example in word_spans]
    # get trigrams in the sentences
    grams = [[[sentence[i:i + 3] for i in range(len(sentence) - 3 + 1)] for sentence in span] for span in spans]
    trigrams_e = [[trigram_dictionary[t] if t in trigram_dictionary else trigram_dictionary['OTH']
                   for e in example for t in e] for example in grams]

# dump the data to a file
# we no longer need the dictionaries
with open("tmp.pickle", "wb") as f:
    pickle.dump((word_embedding_matrix,tag_embedding_matrix, trigrams_e, word_e, tags_e, golden), f)