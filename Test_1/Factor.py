'''
Created on 1 Dec 2015

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
from bcolz.toplevel import arange
import timeit
from sklearn import metrics
#import seaborn as sns
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.scorer import make_scorer
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

df =pd.read_csv('/home/peng/new160half.csv', header=0)
print df
df1=df[df['Target']==1]
df0=df[df['Target']==0]


figsize(10,8)
ax = plt.subplot(111, projection='3d')
ax.scatter(df1['length'], df1['stress'], df1['age'], c='r', s=30, label='Failed')
ax.scatter(df0['length'], df0['stress'], df0['age'], c ='b', marker='^',s=30, label = 'Unfailed')
plt.xlabel('Length of roadway(m)', fontsize=16)
plt.ylabel('Stress factor',fontsize=16)

ax.set_zlabel('Age(years)',fontsize=16, rotation = 90)
plt.legend(fontsize = 16)
#----------------------------- ax.plot(df1['length'], df1['stress'], df1['age'])
#----------------------------- ax.plot(df0['length'], df0['stress'], df0['age'])

plt.show()





