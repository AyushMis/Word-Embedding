import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity
from nltk import word_tokenize

import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    #doc = set('for a of the and to in') # Split into words.
    doc = word_tokenize(doc)
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

#stoplist = set('for a of the and to in'.split())
stoplist = set('for a of the and to in')
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

"""model = Word2Vec(texts, min_count=1, window=5,size=1,workers=1)
word_vectors = model.wv
print (model.similarity('computer','interface'))

print(word_vectors.vocab.keys())"""

from pprint import pprint  # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())

print(new_vec) 
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
for c in corpus:
    print(c)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 2)
corpus_lsi = lsi[corpus_tfidf]
for c in corpus_lsi:
    print(c)
new_vec_lsi = lsi[new_vec]
print(new_vec_lsi)
index = similarities.MatrixSimilarity(lsi[corpus_tfidf])
sims = index[new_vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims)))
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples
train_corpus = list(texts)
model = models.word2vec.Word2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

num_best = 10
instance = WmdSimilarity(corpus, model, num_best=10)
sent = 'The EPS user interface management system'
"""query = preprocess(sent)

sims = instance[query]
print ('Query:')
print (sent)
for i in range(num_best):
    print
    print ('sim = %.4f' % sims[i][1])
    print (documents[sims[i][0]])"""
