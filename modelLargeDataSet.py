import tarfile

filename = 'nips12raw_str602.tgz'
tar = tarfile.open(filename, 'r:gz')
for item in tar:
    tar.extract(item, path='/tmp')

import os, re

# Folder containing all NIPS papers.
data_dir = '/tmp/nipstxt/'  # Set this path to the data on your machine.

# Folders containin individual NIPS papers.
yrs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
dirs = ['nips' + yr for yr in yrs]

# Get all document texts and their corresponding IDs.
docs = []
doc_ids = []
for yr_dir in dirs:
    files = os.listdir(data_dir + yr_dir)  # List of filenames.
    for filen in files:
        # Get document ID.
        (idx1, idx2) = re.search('[0-9]+', filen).span()  # Matches the indexes of the start end end of the ID.
        doc_ids.append(yr_dir[4:] + '_' + str(int(filen[idx1:idx2])))
        
        # Read document text.
        # Note: ignoring characters that cause encoding errors.
        with open(data_dir + yr_dir + '/' + filen, errors='ignore', encoding='utf-8') as fid:
            txt = fid.read()
            
        # Replace any whitespace (newline, tabs, etc.) by a single space.
        txt = re.sub('\s', ' ', txt)
        
        docs.append(txt)
filenames = [data_dir + 'idx/a' + yr + '.txt' for yr in yrs]  # Using the years defined in previous cell.

# Get all author names and their corresponding document IDs.
author2doc = dict()
i = 0
for yr in yrs:
    # The files "a00.txt" and so on contain the author-document mappings.
    filename = data_dir + 'idx/a' + yr + '.txt'
    for line in open(filename, errors='ignore', encoding='utf-8'):
        # Each line corresponds to one author.
        contents = re.split(',', line)
        author_name = (contents[1] + contents[0]).strip()
        # Remove any whitespace to reduce redundant author names.
        author_name = re.sub('\s', '', author_name)
        # Get document IDs for author.
        ids = [c.strip() for c in contents[2:]]
        if not author2doc.get(author_name):
            # This is a new author.
            author2doc[author_name] = []
            i += 1
        
        # Add document IDs to author.
        author2doc[author_name].extend([yr + '_' + id for id in ids])

# Use an integer ID in author2doc, instead of the IDs provided in the NIPS dataset.
# Mapping from ID of document in NIPS datast, to an integer ID.
doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
# Replace NIPS IDs by integer IDs.
for a, a_doc_ids in author2doc.items():
    for i, doc_id in enumerate(a_doc_ids):
        author2doc[a][i] = doc_id_dict[doc_id]

import spacy
nlp = spacy.load('en')


processed_docs = []    
for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Remove common words from a stopword list.
    #doc = [token for token in doc if token not in STOPWORDS]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)

docs = processed_docs
del processed_docs

from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

from gensim.corpora import Dictionary
dictionary = Dictionary(docs)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.

corpus = [dictionary.doc2bow(doc) for doc in docs]
print('Number of authors: %d' % len(author2doc))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

from gensim.models import AuthorTopicModel
model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, \
                author2doc=author2doc, chunksize=2000, passes=1, eval_every=0, \
                iterations=1, random_state=1)

model_list = []
for i in range(5):
    model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, \
                    author2doc=author2doc, chunksize=2000, passes=100, gamma_threshold=1e-10, \
                    eval_every=0, iterations=1, random_state=i)
    top_topics = model.top_topics(corpus)
    tc = sum([t[1] for t in top_topics])
    model_list.append((model, tc))


model, tc = max(model_list, key=lambda x: x[1])
print('Topic coherence: %.3e' %tc)

model.save('/tmp/model.atmodel')
model = AuthorTopicModel.load('/tmp/model.atmodel')
model.show_topic(0)
topic_labels = ['Circuits', 'Neuroscience', 'Numerical optimization', 'Object recognition', \
               'Math/general', 'Robotics', 'Character recognition', \
                'Reinforcement learning', 'Speech recognition', 'Bayesian modelling']
for topic in model.show_topics(num_topics=10):
    print('Label: ' + topic_labels[topic[0]])
    words = ''
    for word, prob in model.show_topic(topic[0]):
        words += word + ' '
    print('Words: ' + words)
    print()






















































