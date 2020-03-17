import os
import string
import nltk
from nltk.corpus import gutenberg
import pandas as pd

from gensim import corpora, models
from collections import defaultdict

from urllib import request


def tokenize(doc):

    stopwords = nltk.corpus.stopwords.words('english')
    tokens = doc.split()
    tokens = [t for t in tokens if t.lower() not in stopwords]
    tokens = [t.lower() for t in tokens if t.isalpha()]

    return tokens


if not os.path.exists('shakespeare-complete-raw.txt'):
    print('downloading shakespeare text...')
    url = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    with open('shakespeare-complete-raw.txt', 'w') as o:
        o.write(raw)

print('processing text...')
text = open('shakespeare-complete-raw.txt').read()
sonnets_complete = text[text.find('THE SONNETS'):text.find('THE END')]
sonnets = sonnets_complete.split('\n\n\n')
sonnets = [tokenize(s) for s in sonnets]
sonnets = sonnets[1:-1]  # remove the first and last

dictionary = corpora.Dictionary(sonnets)
# dictionary.filter_extremes(no_above=0.25)
dictionary.filter_n_most_frequent(100)

corpus = [dictionary.doc2bow(sonnet) for sonnet in sonnets]

print('training model...')
model = models.LdaModel(corpus, id2word=dictionary, num_topics=20)
print(model.print_topics())
