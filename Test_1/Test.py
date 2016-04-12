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


df=pd.DataFrame({'x':np.arange(0,10,1),'y':np.arange(0,10,1)})
x=[]
x.append(df.iloc[0:5].as_matrix())
print x
print x[0]

