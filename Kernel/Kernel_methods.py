'''
Created on 15 Feb 2016

@author: peng
'''
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from scipy.interpolate import spline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
import timeit
from mpl_toolkits.mplot3d import Axes3D
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


start = timeit.default_timer()

####### Read the source data######################
df =pd.read_csv('Source_Data.csv', header=0)

p= Preprocessdata.standardprocess()



train, trainlabel, test, testlabel = p.scaledivd(df, 1.0)
print np.shape(train)

###################################### PCA  #############################
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
newtrain=pca.fit_transform(train)
print pca.explained_variance_ratio_ 
print np.sum(pca.explained_variance_ratio_)
print np.shape(newtrain)


train = newtrain



##########################################################################################################################

###############################################Train the model

ff = mysvc.training_manCV()

df = ff.trainSVC(train, trainlabel, 'poly', Cmin=-10, Cmax=10, numC=21, rmin=-10, rmax=10, numr=21, degree = 4)

df.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_pca6_cm_10CV_d4_n10_p10_21.csv', header = True)



################################################ ####


####################### Read the cm and convert cm to metrics########



#===============================================================================
# df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/linear_cm_10CV_n10_p10_21.csv', header = 0)
# 
# 
# #### transform the raw data into accuracy or precision
# 
# df1 = df.drop('gamma_range',axis=1)
# df2 = df1.drop('Unnamed: 0',axis=1)
#  
# df_assess = DataFrame()
# for i in df2.columns.values:
#     df_assess[i]= ff.accuracy( ff.str_float(df2[i]))
# 
# df_assess.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/linear_acc_10CV_n10_p10_21.csv', header = True)
#===============================================================================

#####        

###### Plotting the gridsearch##########


#df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_acc_10CV_d3_n10_p10_21.csv', header = 0)

#===============================================================================
# scores=np.array(df_assess)
# scores=scores[:, :].T
# #    print scores
# #scores= scores[:,5:]
# print np.shape(scores)
# 
# #    print np.arange(100,2010,20)
# 
# figsize(8,6.5)
# fig, ax = plt.subplots(1,1)
# cax = ax.imshow(scores, interpolation='none', origin='highest',
#                 cmap=plt.cm.coolwarm, aspect=1)
# 
# plt.grid(b=False, which='x', color='white',linestyle='-')
# 
# 
# 
# 
# plt.xticks(np.arange(-0.5,21.5,5), (-10,-5,0,5,10),fontsize = 20)
# plt.yticks(np.arange(-0.5,23.5,5), (-10,-5,0,5,10),fontsize = 20)
# 
# #plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
# 
# plt.ylabel('$log_2 C$',fontsize = 24)
# plt.xlabel('$log_2 \gamma$', fontsize = 24)
# 
# ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.grid(b=True, which='major', color='w', linewidth=0.5)
# ax.grid(b=True, which='minor', color='w', linewidth=0.5)
# 
# 
# cb = fig.colorbar(cax)
# cb.ax.tick_params(labelsize=14)
# 
# 
# plt.show()
#===============================================================================

#######plot the acc and prec of linear kernel###########



#===============================================================================
# figsize(9.5,8)
#  
# df1 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/linear_acc_10CV_n10_p10_21.csv', header=None)
# df2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/linear_prec_10CV_n10_p10_21.csv', header=None)
#  
# df1=df1.T
# df2=df2.T
# 
# x_axis=np.arange(-10,11,1)
# x_axis=x_axis
# x_new = np.linspace(x_axis.min(), x_axis.max(), 50)
# 
# list_df1=np.array(df1[1][1:])
# print list_df1
# list_df1=list_df1.T
# print list_df1
# 
# print np.shape(list_df1)
# #power_line = spline(x_axis, list_df1, x_new)
# plt.plot(x_axis, df1[1][1:],label='Accuracy', color='red')
# plt.plot(x_axis, df2[1][1:], label = 'Precision',color='blue')
# 
# 
# plt.scatter(x_axis, df1[1][1:], label=None, color='red')
# #plt.plot(x_new, power_line, label='Accuracy')
# plt.scatter(x_axis, df2[1][1:], label=None,color='blue')
# plt.xlim(-10,10)
# plt.legend(fontsize = 20)
# #plt.xticks(np.arange(0,20,5), (-10,-5,0,5,10),fontsize = 20)
# plt.xticks(fontsize =20)
# plt.yticks(fontsize =20)
# plt.ylabel('Scores', fontsize = 24)
# plt.xlabel('$log_2 C$', fontsize = 24)
# plt.show()
#===============================================================================


####################################





stop = timeit.default_timer()

print "The running takes %r min" %((stop-start)/60)