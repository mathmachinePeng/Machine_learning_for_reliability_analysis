'''
Created on 10 Jun 2016

@author: peng
'''
#0. initialize the dataset, get the size
import pandas as pd
import numpy as np

data_path = '/home/peng/Documents/NLP/Kaggle_datasets/'

df = pd.read_csv(data_path + 'labeledTrainData.tsv', header = 0, \
                    delimiter = '\t', quoting = 3)
num_docus = train['review'].size


#1. remove the HTML markup( like <br>), remove non-letters, convert to lower case, split into 
# words, remove stopwords, join words back into one string separated by space

import Preprocessing_nlp as pre
clean_docus = []
for i in xrange(0, num_docus):
    
    if ((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_docus)
    
    clean_docus.append(pre.review_to_words(df['clean_url'][i], filter_words = 'timeline'))

# 1.2 further filtering more words (optional)


    
#####################################################
#2.1    create features from the bag of words
print 'creating the bag of words...\n'
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer( analyzer = 'word', \
                            tokenizer = None, \
                            preprocessor = None, \
                            stop_words = None, \
                            max_features = 5000)
'''
The feature should be the size of features
To limit the size of the feature vectors, we should choose some maximum vocabulary size. Below, we use the 5000 most 
frequent words (remembering that stop words have already been removed).
'''


train_data_features = vectorizer.fit_transform(clean_docus)

train_data_features = train_data_features.toarray()

print train_data_features.shape
