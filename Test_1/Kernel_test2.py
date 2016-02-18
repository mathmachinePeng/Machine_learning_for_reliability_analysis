'''
Created on 18 Feb 2016

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

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import re
from astropy.io.fits.header import Header


#secret_cm = []



start = timeit.default_timer()
df =pd.read_csv('Source_Data.csv', header=0)

p= Preprocessdata.standardprocess()


#    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/score50-500_2.csv', header=0)   
train, trainlabel, test, testlabel = p.scaledivd(df, 1.0)
print np.shape(train)

#C_range=np.logspace(1, 2, num=5, base=2)  
#gamma_range=np.logspace(1, 2, num=5, base=2) 
C_range=np.logspace(-10, 10, num=21, base=2,endpoint= True)
gamma_range=np.logspace(-10, 10, num=21, base=2,endpoint= True)

###############################################

ff = mysvc.training_manCV()

df = ff.trainSVC(train, trainlabel, 'poly', Cmin=-10, Cmax=10, numC=21, rmin=-10, rmax=10, numr=21, degree = 4)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------- print df, df_this
#------------------------------------------------------------------------------ 
df.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_cm_10CV_d4_n10_p10_21.csv', header = True)
# df_this.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/Try_this_score.csv', header = True)


################################################

#df.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/TRYleft_rbf_n10_p10_21.csv', header = True)

#######################



#df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_cm_10CV_d3_n10_p10_21.csv', header = 0)
#print df

#### transform the raw data into accuracy or precision

#------------------------------------------- df1 = df.drop('gamma_range',axis=1)
#------------------------------------------- df2 = df1.drop('Unnamed: 0',axis=1)
#------------------------------------------------------------------------------ 
#------------------------------------------------------- df_assess = DataFrame()
#-------------------------------------------------- for i in df2.columns.values:
    #------------------------- df_assess[i]= ff.precision( ff.str_float(df2[i]))
#------------------------------------------------------------------------------ 
# df_assess.to_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_prec_10CV_d3_n10_p10_21.csv', header = True)

#####        

###### Plotting the gridsearch##########


#df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/poly_acc_10CV_d3_n10_p10_21.csv', header = 0)

#---------------------------------------------------- scores=np.array(df_assess)
#--------------------------------------------------------- scores=scores[:, :].T
#------------------------------------------------------------- #    print scores
#--------------------------------------------------------- #scores= scores[:,5:]
#-------------------------------------------------------- print np.shape(scores)
#------------------------------------------------------------------------------ 
#--------------------------------------------- #    print np.arange(100,2010,20)
#------------------------------------------------------------------------------ 
#---------------------------------------------------------------- figsize(8,6.5)
#--------------------------------------------------- fig, ax = plt.subplots(1,1)
#--------------- cax = ax.imshow(scores, interpolation='none', origin='highest',
                #------------------------------- cmap=plt.cm.coolwarm, aspect=1)
#------------------------------------------------------------------------------ 
#--------------------- plt.grid(b=False, which='x', color='white',linestyle='-')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------- plt.xticks(np.arange(-0.5,21.5,5), (-10,-5,0,5,10),fontsize = 20)
#------------- plt.yticks(np.arange(-0.5,23.5,5), (-10,-5,0,5,10),fontsize = 20)
#------------------------------------------------------------------------------ 
#-------------- #plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
#------------------------------------------------------------------------------ 
#----------------------------------------- plt.ylabel('$log_2 C$',fontsize = 24)
#----------------------------------- plt.xlabel('$log_2 \gamma$', fontsize = 24)
#------------------------------------------------------------------------------ 
#--------------- ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
#--------------- ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
#---------------------- ax.grid(b=True, which='major', color='w', linewidth=0.5)
#---------------------- ax.grid(b=True, which='minor', color='w', linewidth=0.5)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------------- cb = fig.colorbar(cax)
#----------------------------------------------- cb.ax.tick_params(labelsize=14)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------------- plt.show()



####################################



stop = timeit.default_timer()
print "The running takes %r min" %((stop-start)/60)