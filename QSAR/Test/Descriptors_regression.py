'''
Created on 29 Sep 2015

@author: peng
'''
from IPython.core.pylabtools import figsize
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import RFclass
import Preprocessdata 
#import Preprocessdata1 as p
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV
import matplotlib as mpl
from scipy.interpolate import spline

import timeit
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.scorer import make_scorer
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
# process data into scaled training and testing



# cls = ['rf', 'adb', 'bag', 'ext', 'gbt']


start = timeit.default_timer()
    
'''
:: Read the data
'''    
    
df =pd.read_csv('QSAR_inhibitor_eff.csv', header=0)

    
    
'''
Preprocess the data
'''    

p= Preprocessdata.standardprocess()


train, trainlabel, test, testlabel =p.noscale(df,0.8)

print np.shape(train)

ff=RFclass.training()
tt = RFclass.test()

forest = ff.trainforest('rf_regress', train, trainlabel, 500, 11)
r2, rmse = tt.testforest_R(test, testlabel, forest)
print r2, rmse    
    
    



stop = timeit.default_timer()
print "The running takes %r min" %((stop-start)/60)

    
    
   
    


    
    
    
    
    
 






