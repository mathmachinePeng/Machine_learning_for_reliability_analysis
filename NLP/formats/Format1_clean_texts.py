'''
Created on 10 Jun 2016

@author: peng
'''
#0. initialize the dataset, get the size
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np

def review_to_words( raw_review, filter_words = None ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
#    letters_only = re.sub(filt_words, " ", review_text) 
    
    
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()  
                     
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english")) 
    stops.add(filter_words)                 
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  

data_path = '/home/peng/Documents/NLP/Kaggle_datasets/'

df = pd.read_csv(data_path + 'labeledTrainData.tsv', header = 0, \
                    delimiter = '\t', quoting = 3)
num_docus = df['review'].size


#1. Remove the url linkage
clean_url = []
noise = []  # the index of those missing values
length = len(df['text'])
for i in xrange(0, length):     
    if type(df['text'][i]) == unicode:
        review_text = BeautifulSoup(df['text'][i]).get_text() 
        URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', review_text)
        clean_url.append(URLless_string)
        if i%1000==0:
            print 'already %d has finished in %d' %(i, length)
    else:
        clean_url.append('0')
        noise.append(i)
               

#2. remove the HTML markup( like <br>), remove non-letters, convert to lower case, split into 
# words, remove stopwords, join words back into one string separated by space


clean_docus = []
for i in xrange(0, num_docus):
    
    if ((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_docus)
    
    clean_docus.append(review_to_words(df['clean_url'][i], filter_words = 'timeline'))




    
#####################################################
#3.1    create features from the bag of words
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
