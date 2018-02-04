from __future__ import unicode_literals
import array
from collections import defaultdict
import io
import logging
import os

import six
from six.moves.urllib.request import urlretrieve
import torch
from tqdm import tqdm
import tarfile
import numpy as np
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class Vectors(object):

    def __init__(self, name, cache='.vector_cache',
                 url=None, unk_init=torch.Tensor.zero_):
        """Arguments:
               name: name of the file that contains the vectors
               cache: directory for cached vectors
               url: url for download if vectors not found in cache
               unk_init (callback): by default, initalize out-of-vocabulary word vectors
                   to zero vectors; can be any function that takes in a Tensor and
                   returns a Tensor of the same size
         """
        self.unk_init = unk_init
        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None):
        path = os.path.join(cache, name)
        path_pt = path + '.pt'

        if not os.path.isfile(path_pt):
            itos, vectors, dim = [], array.array(str('d')), None
            path = "glove.840B1.300d.txt" 
                # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
                # If there are malformed lines, read in binary mode
                # and manually decode each word from utf-8
            except:
                #logger.warning("Could not read {} as UTF8 file, "
                #               "reading file as bytes and skipping "
                #               "words with malformed UTF8.".format(path))                with open(path, 'rb') as f:
                with open(path,'rb') as f:
                    lines = [line for line in f]
                binary_lines = True
                
            logger.info("Loading vectors from {}".format(path))
            for line in tqdm(lines, total=len(lines)):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(" ")
                
                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue
                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
                #print('tensor vector: {}'.format(self.vectors))
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)


class GloVe(Vectors):
    url = {
        '840B1': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B1', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)
        print(super(GloVe, self).__init__(name, url=url, **kwargs))

def get_word(word):
    return glove.vectors[glove.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))
        
def listing(tuple):
    temp = []
    for element in tuple:
        temp.append(element[0])
    return temp


def _default_unk_index():
    return 0


pretrained_aliases = {
    #"glove.6B.50d": lambda: GloVe(name="6B", dim="50"),
    #"glove.6B.100d": lambda: GloVe(name="6B", dim="100"),
    #"glove.6B.200d": lambda: GloVe(name="6B", dim="200"),
    #"glove.6B.300d": lambda: GloVe(name="6B", dim="300")
    "glove.840B1.300d": lambda: GloVe(name="840B", dim="300")
}

glove = GloVe(name='840B1', dim=300)

print('Loaded {} words'.format(len(glove.itos)))

#Target word
target_word = input("Enter the target word: ")

num = int(input("Enter the no. of sentences which you want to add: "))

sentences = []

#Input set of sentences containing target word
for i in range(num):
    sentences.append(input("Enter your sentence: ").split())

#Stop words as defined in NLTK library
stop_words = stopwords.words('english')

#Filtering out stopwords from sentence
sentences = [[word for word in sentence if ((word not in stop_words) and word.isalpha()) ]
         for sentence in sentences]
print(sentences)

#Developing a vector defining each sentence
vectors=[]

for sentence in sentences:
    sum1 = get_word(sentence[0])
    for i in range(1,len(sentence)):
        sum1=sum1+get_word(sentence[i])
    sum1 = sum1/len(sentence)
    vectors.append(sum1 + get_word(target_word)) #Here we are giving some extra weight to our target word.
                                                 #This is where we have to find some proper weightage to be
                                                 #given to contxt and target word individually
i=0
s = []
for vector in vectors:
    #print("Similar words for sentence: ",i)
    s.append(vector + get_word(target_word))
    i+=1
    #print_tuples(closest(vector + get_word(target_word))) #Adding second term of target_word for giving more weightage to the target wordas compared to context words.

common_vector = vectors[0]

#Evaluating a vector for whole set of sentences
if len(s)>1:
    for i in range(1,len(s)):
        common_vector+=vectors[i]

    
print("Common Similar words to context: ")
print_tuples(closest(common_vector))

common = closest(common_vector)
target_sim = closest(get_word(target_word))

common_words = listing(common)
target_words = listing(target_sim)

final = []
for word in target_words:
    if word in common_words:
        for word2, i in target_sim:
            if word == word2:
                prob = i
                final.append((word, prob))

print("Final list of similar words with their probabilities: ")
print_tuples(final)
