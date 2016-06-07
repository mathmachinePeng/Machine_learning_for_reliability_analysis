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

'''
Here, "header=0" indicates that the first line of the file contains column names, 
"delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells 
Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.
'''
#print train.shape
#print train.columns.values
##print train['review'][0]
###############################

#------------------------------------------------------------------------------ 
#--- #### Noted, Section 2 has been organized into a methdo in Preprocessing_nlp
#------------------------------------------------------------------------------ 
#----------------------------------- ### 2. Data cleaning and Text preprocessing
#------------------------------------------------------------------------------ 
#----------------------- ### 2.1 Removing HTML Markup: The BeautifulSoup Package
#------------------------------------------------- from bs4 import BeautifulSoup
#---------------- # Initialize the BeautifulSoup object on a single movie review
#---------------------------------- example1 = BeautifulSoup(train['review'][0])
#------------------------------------------------------------------------------ 
#----------------------------------------------------- #print train["review"][1]
#---------------------------------------------------- #print example1.get_text()
#------------------------------------------------------------------------------ 
# ### 2.2 Dealing with Punctuation, Numbers and Stopwords: NLTK and regular expressions
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------------- import re
#---------------------------- # Use regular expressions to do a find-and-replace
#------ letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      #- " ",                   # The pattern to replace it with
                      #------------- example1.get_text() )  # The text to search
#--------------------------------------------------------------------------- '''
 # [] indicates group membership and ^ means "not".  In other words, the re.sub() statement above
  # says, "Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and
  #--------------------------------------------------- replace it with a space."
#--------------------------------------------------------------------------- '''
#----------------------------------------------------------- #print letters_only
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------ ######
#------------------------------------------------------------------------------ 
#--------- ### 2.3 Tokenization: convert samples into single lower case words###
#------------------------------------------------------------------------------ 
#-------------- lower_case = letters_only.lower()        # Convert to lower case
#------------------- words = lower_case.split()               # Split into words
#------------------------------------------------------------------------------ 
#------------------------------------------------- ### 2.4 remove the stop words
#------------------------------------------------------------------- import nltk
#--------------------------------------------- from nltk.corpus import stopwords
#---------------------------------------------- print stopwords.words('english')
#------------------------------------------------------------------------------ 
#------------- words = [w for w in words if not w in stopwords.words("english")]
#------------------------------------------------------------------- print words
###########################################################################

#------------------- ### using method to realize the function of above section 2
#------------------------ clean_review = pre.review_to_words(train['review'][0])
#------------------------------------------------------------ print clean_review


### loop and clean all the training set at once
num_reviews = train['review'].size
clean_train_reviews = []

for i in xrange(0, num_reviews):
    
    if ((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_reviews)
    
    clean_train_reviews.append(pre.review_to_words(train['review'][i]))
    
##### 3. Creating features from a Bag of Words
'''
 The Bag of Words model learns a vocabulary from all of the documents, then models each 
 document by counting the number of times each word appears.
'''

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer
# 3.1 fit the countvectorizer model with training set
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()







