from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import RFclass
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
import TAlogistic as tl
import cPickle, theano
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import TAmlp as mlp


import TAdbn as dbn
import TAsda as sda
from scipy.interpolate import spline


y=[]
for i in np.arange(2):
    y.append(i+2)
c = np.mean(y)    
print y,c
#print df_2[df_2['gbt']== df_2['gbt'].max()]
#----------- feautre_2 = [0.812500,   0.770833,   0.812500,   0.791667, 0.8125 ]
#-------- feature_4 = [0.812500,   0.770833,   0.791667,   0.812500,  0.833333 ]
#-------- feature_6 = [0.812500,   0.770833,   0.812500,   0.833333,  0.812500 ]
#-------- feature_8 = [0.812500,   0.770833,   0.791667,   0.812500,  0.812500 ]
#--------------- cls = [ 'rf',        'adb' ,       'bag'    ,    'ext' , 'gbt']