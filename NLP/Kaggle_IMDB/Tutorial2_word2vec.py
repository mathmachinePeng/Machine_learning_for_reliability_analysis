'''
Created on 6 Jun 2016

@author: peng
'''
import pandas as pd
import Preprocessing_nlp as pre
#from NLP.Kaggle_IMDB.Preprocessing_nlp import review_to_words

###  1. import the data###

data_path = '/home/peng/Documents/NLP/Kaggle_datasets/'

train = pd.read_csv(data_path + 'labeledTrainData.tsv', header = 0, \
                    delimiter = '\t', quoting = 3)
test = pd.read_csv(data_path + 'testData.tsv', header = 0, \
                    delimiter = '\t', quoting = 3)
unlabeled_train = pd.read_csv(data_path + 'unlabeledTrainData.tsv', header = 0, \
                    delimiter = '\t', quoting = 3)

#verify the shape
print 'Read %d labeled train reviews, %d labeled test reviews, ' \
' and %d unlabeled reviews\n' %(train['review'].size, test['review'].size, unlabeled_train['review'].size)

## 2. preprocess the data
import Preprocessing_nlp as pre
'''
:word2vec does not need remove stop words,  because the algorithm relies on the broader 
context of the sentence in order to produce high-quality word vectors.

:word2vec expects single sentences, each one as a list of words a list of lists
'''
## 2.1  use NLTK's punkt tokenizer for sentence splitting
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

print 'Parsing sentences from training set'
for review in train['review']:
    sentences += pre.review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += pre.review_to_sentences(review, tokenizer)
    
    
print len(sentences)
print sentences[0]