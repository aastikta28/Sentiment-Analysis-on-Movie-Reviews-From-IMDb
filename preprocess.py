# -*- coding: utf-8 -*-
#import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import os
import glob
#from pprint import pprint

#sys.path.append("/")
labels = open('5k_spring_2016_label_training.txt', 'w')
directory = "5k_dataset/train/neg"
#directory = "sample_train_set"
neg = []
docs = []
for file_path in glob.glob(os.path.join(directory, "*")):
    neg_file = open(file_path).read()
    neg.append(neg_file)
    docs.append(neg_file)
    labels.write('-1')
    labels.write('\n')
    #print "neg here"
        
directory = "5k_dataset/train/pos"
#directory = "sample_train_set/pos"
pos = []
for file_path in glob.glob(os.path.join(directory, "*")):
    pos_file = open(file_path).read()
    pos.append(pos_file)
    docs.append(pos_file)
    labels.write('1')
    labels.write('\n')
    #print "here"
    
text = []

en_stop = stopwords.words('english')

tokenizer = RegexpTokenizer(r'\w+')

p_stemmer = PorterStemmer()

for i in docs:
        
        # clean and tokenize document string
        raw = i.lower()
        #raw = [line.lower() for line in i]
        tokens = tokenizer.tokenize(raw)
        #print(tokens)
    
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        
        #print(stemmed_tokens)
        # add tokens to list
        text.append(stemmed_tokens)

#pprint(text)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(text)
#print dictionary
#print(dictionary.token2id)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(t) for t in text]
#print(corpus)

f = open('5k_spring_2016_training_dataset.txt', 'w')
for doc in range(0 , len(corpus)):
    for k,v in corpus[doc]:
        line = str(doc+1) + " " + str(k) + " " + str(v)
        f.write(line)
        f.write('\n')
    
labels.close()
f.close()