'''
Created on 6 Jan 2016

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
#import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import re
import Superplot
from sklearn import svm
from sklearn.metrics import recall_score, precision_score, r2_score
import MySVM


da= pd.read_csv('/home/peng/dataset/Lin_viscosity_v3_random.csv',header=0)



#===============================================================================
# """data preprocessing"""
# da.fillna(0, inplace=True)
# daa = da.sample(61)
# daa.to_csv('/home/peng/dataset/Lin_viscosity_v3_random.csv',header=True)
#===============================================================================


cols = da.columns.tolist()
print cols
#cols=['CaO', 'SiO2', 'Al2O3', 'B2O3', 'Na2O', 'TiO2', 'MgO', 'Li2O', 'MnO', 'ZrO2','xT4+','xT3+','xM+','xM2+','xM4+','a']
cols=['CaO', 'SiO2', 'Al2O3', 'B2O3', 'Na2O', 'TiO2', 'MgO', 'Li2O', 'ZrO2','xT4+','xT3+','xM+','xM2+','a']
df= da[cols]
 
p= Preprocessdata.standardprocess()
train, trainlabel, test, testlabel =p.noscale(df,0.7)



#===============================================================================
# ff= RFclass.training()
# forest=ff.trainforest('rf_regress', train, trainlabel, number_trees=500, number_features=2)
# tt = RFclass.test()
#  
# r2= tt.testforest_R(test, testlabel, forest)
# print r2
#===============================================================================
 





####################SVM##############
#===============================================================================
# svrtry = MySVM.training_regress()
# #svrtry.svmlinear(train, trainlabel, -20, 20, 50)
# model, scores = svrtry.svmrbf(train, trainlabel, Cmin=-10, Cmax=10, gmin=-10, gmax=10, num=10)
# output = model.predict(test)
# r2 = r2_score(testlabel, output)
# print r2
#===============================================================================


from sklearn import linear_model
clf = linear_model.BayesianRidge()
clf.fit(train, trainlabel)
output = clf.predict(test)
print output
r2 = r2_score(testlabel, output)
print r2

