import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.spatial import distance
import random
from collections import Counter 

lemmatizer = WordNetLemmatizer()

def create_lexicon(sample):
	lexicon = []
	stop_words = set(stopwords.words('english'))
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:100]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)
		lexicon=[word.lower() for word in lexicon if word.isalpha()]
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	lexicon = [w for w in lexicon if not w in stop_words]
	w_counts = Counter(lexicon)
	vocab = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
		    vocab.append(w)
	return vocab, lexicon

vocab, corpus = create_lexicon('enwik8')
vocab_size = len(vocab)
print (vocab)
print (corpus)

word2int = {}
int2word = {}
for i,word in enumerate(vocab):
    word2int[word] = i
    int2word[i] = word

WINDOW_SIZE = 2

data = []

for word_index, word in enumerate(corpus):
	for nb_word in corpus[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(corpus)) + 1] : 
		if nb_word != word:
			data.append([word, nb_word])

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!


# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))


# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000
# train for n_iter iterations

for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    #print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)

"""def find_closest(word_index, vectors):
	min_dist = 10000 # to act like positive infinity
	min_index = -1
	query_vector = vectors[word_index]
	for index, vector in enumerate(vectors):
		if distance.euclidean(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
		min_dist = distance.euclidean(vector, query_vector)
		min_index = index
	return min_index"""

def find_closest(word_index, vectors):
	min_dist = 10000 # to act like positive infinity
	min_index = -1
	query_vector = vectors[word_index]
	for index, vector in enumerate(vectors):
		if distance.euclidean(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
			min_dist = distance.euclidean(vector, query_vector)
			min_index = index
			#storage.append(min_index)
	return min_index

from sklearn import preprocessing
import matplotlib.pyplot as plt
vectors = preprocessing.normalize(vectors, norm='l2')

fig, ax = plt.subplots()
print(word)
for word in vocab:
    #print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()
