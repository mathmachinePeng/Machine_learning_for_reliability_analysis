'''
Created on 4 Jul 2016
This interface class if for visualizing results of random forest-based model 

@author: peng
'''
from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier as RFC


from bayes_opt import BayesianOptimization
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from scipy.interpolate import spline
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.gaussian_process import GaussianProcess
import timeit

import seaborn as sns
from IPython.core.pylabtools import figsize
from scipy.interpolate import spline
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import re
from astropy.io.fits.header import Header
from matplotlib.pyplot import xlim
import sys
sys.path.append('/home/peng/git/Machine_learning_for_reliability_analysis/Preprocess')
import Preprocessdata
import RFclass

## default setting
data_path = '../Data/'
p= Preprocessdata.standardprocess()

# class
class MyClass(object):

    def __init__(self, Training_data):
        
        df =pd.read_csv(data_path + Training_data, header=0)
        self.train, self.trainlabel, self.test, self.testlabel = p.noscale(df, 1.0)
        print ('This dataset contains %r samples with %r features' %(np.shape(self.train)[0], np.shape(self.train)[1]))
        print ('These features are:')
        for i in xrange(0, np.shape(self.train)[1]):
            print ('Feature %r : %r' %(i+1, list(df.columns.values)[i]))
       
    def training(self):
        bestmodel= RFC(n_estimators= 100).fit(self.train,self.trainlabel)   
        ff= RFclass.training()
        ff.importance(bestmodel, 12, color = '#66cdaa', plot_std = False)
        return bestmodel  
    
    def testing(self, bestmodel, Test_data):
        
        df_1 =pd.read_csv(data_path + Test_data, header=0)
        train_, trainlabel_, test_, testlabel_ = p.noscale(df_1, 0.8)
        proba =  bestmodel.predict_proba(test_)
        for i in xrange(0,len(test_)):
            print ('Test sample %r has a probability of failure %0.2f' % (i+1, proba[i,1]))
          