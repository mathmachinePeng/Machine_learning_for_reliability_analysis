'''
Created on 6 Jun 2016

@author: peng
'''
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

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    
    #3.5 remove more words
#    review_text = re.sub(filter_words, " ", review_text)
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def makeFeatureVec(words, model, num_features):
## this function is for averaging all word vectors in a given paragraph

# pre-initialize an empty numpy array for speed
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0
    
    # index =2word is a list that contains the names of the words in vocabulary, 
    #convert to set for speed    
    index2word_set = set(model.index2word)
    
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
            
    featureVec = np.divide(featureVec, nwords)
    
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate the average feature vectors for
    #each one and return a 2d numpy array
    
    counter = 0
    #preallocate a 2D numpy array for spped
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = 'float32')
    
    for review in reviews:
        if  counter%1000 ==0:
            print 'review %d of %d' % (counter, len(reviews))
            
        #call the function that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        
        counter = counter +1
        
    return reviewFeatureVecs        
        
 
######
'''
 The function above will give us a numpy array for each review, each with a number of 
 features equal to the number of clusters. Finally, we create bags of centroids for our
  training and test set, then train a random forest and extract results:
'''
        
def create_bag_of_centroids(wordlist, word_centroid_map):
    
    
    
    # the number of clusters equal to the highest cluster index
    
    num_centroids = max(word_centroid_map.values()) + 1
    
    # pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros ( num_centroids, dtype = 'float32')
    
    #loop over the words in the review. If the word is in the vocabulary, find which clusters
    # it belongs to, and increment that cluster count by one
    
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids
          
    

