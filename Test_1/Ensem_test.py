'''
Created on 5 Nov 2015

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

#------------------------------------------------ start = timeit.default_timer()
#------------------------------------------------------------------------------ 
# df =pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/adb_100_4000_10times.csv', header = 0)
#------------------------------------------------------------------------------ 
#----------------------------------------------------- fig, ax1 = plt.subplots()
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#--------- ax1.plot(df['Tree'], pd.rolling_mean(df['Score'], window = 30), 'b-')
#------------------------------------------------------- ax1.set_xlabel('Trees')
#----------------- # Make the y-axis label and tick labels match the line color.
#------------------------------------- ax1.set_ylabel('Rolling mean', color='b')
#---------------------------------------------- for tl in ax1.get_yticklabels():
    #--------------------------------------------------------- tl.set_color('b')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------- ax2 = ax1.twinx()
#---------- ax2.plot(df['Tree'], pd.rolling_std(df['Score'], window = 10), 'r-')
#-------------------------------------- ax2.set_ylabel('Rolling std', color='r')
#---------------------------------------------- for tl in ax2.get_yticklabels():
    #--------------------------------------------------------- tl.set_color('r')
#-------------------------------------------------------------------- plt.show()
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------- stop = timeit.default_timer()
#--------------------------- print "The running takes %r min" %((stop-start)/60)

    
        
ff = RFclass.training()
tt = RFclass.test()
pp = Superplot.fancy()
#---------------------------------------------------------------- figsize(9.5,7)
#------------------------------------------------------------------------------ 
# df1 = pd.read_csv('//home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/bag_acc_10cv_100_4000.csv', header=0)
# df2 = pd.read_csv('//home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/bag_prec_10cv_100_4000.csv', header=0)
#---------------------- plt.plot(df1['tree_range'], df1['12'], label='Accuracy')
#------------------- plt.plot(df1['tree_range'], df2['12'], label = 'Precision')
#----------------------------------------------------- plt.legend(fontsize = 20)
#------------------------------------------------------ plt.xticks(fontsize =20)
#------------------------------------------------------ plt.yticks(fontsize =20)
#--------------------------- plt.ylabel('Classification metrics', fontsize = 24)
#---------------------------------- plt.xlabel('Number of trees', fontsize = 24)
#-------------------------------------------------------------------- plt.show()


##########################plot ##########################
# df1= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/bag_acc_63_100_4000.csv', header=0)
#------------------------------------------------------------------------------ 
# df2= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/bag_prec_63_100_4000.csv', header=0)
#------------------------------------------------------------------------------ 
# #df3= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/bag_acc_63_f12_t1000_100times.csv', header=0)
#------------------------------------------------------------------------------ 
# #df4= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/bag_prec_63_f12_t1000_100times.csv', header=0)
#----------------------------------------------------------------- figsize(10,8)
# plt.plot(df1['tree_range'][0:96], df1['12'][0:96], label = '66:33 splitting accuracy')
# plt.plot(df1['tree_range'][0:96], df2['12'][0:96], label = '66:33 splitting precision')
#---------------------------------- plt.xlabel('Number of trees', fontsize = 24)
#--------------------------- plt.ylabel('Classification metrics', fontsize = 24)
#----------------------------------------------------- plt.xticks(fontsize = 20)
#----------------------------------------------------- plt.yticks(fontsize = 20)
#----------------------------------------------------- plt.legend(fontsize = 20)
#----------- #plt.plot(df1['times'], df3['acc_score'], label = '66:33 accuracy')
#---------- #plt.plot(df1['times'], df4['prec_score'], label = '66:33 accuracy')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------------- plt.show()


################################# plot sensitivity ######################################################

# df1= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/rf_acc_63_f11_t800_100times.csv', header=0)
#------------------------------------------------------------------------------ 
# df2= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/rf_prec_63_f11_t800_100times.csv', header=0)
#------------------------------------------------------------------------------ 
 # #df3= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/bag_acc_63_f12_t1000_100times.csv', header=0)
#------------------------------------------------------------------------------ 
 # #df4= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/bag_prec_63_f12_t1000_100times.csv', header=0)
#----------------------------------------------------------------- figsize(10,8)
#------------ plt.plot(df1['times'], df1['acc_score'], label = '66:33 accuracy')
#---------- plt.plot(df1['times'], df2['prec_score'], label = '66:33 precision')
#-------------------------------------------- plt.xlabel('Times', fontsize = 24)
#--------------------------- plt.ylabel('Classification metrics', fontsize = 24)
#----------------------------------------------------- plt.xticks(fontsize = 20)
#----------------------------------------------------- plt.yticks(fontsize = 20)
#----------------------------------------------------- plt.legend(fontsize = 20)
#----------- #plt.plot(df1['times'], df3['acc_score'], label = '66:33 accuracy')
#---------- #plt.plot(df1['times'], df4['prec_score'], label = '66:33 accuracy')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------------- plt.show()


df1= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/adb_acc_10cv_f12_t700_100times.csv', header=0)
#------------------------------------------------------------------------------ 
df2= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/Results/statistical_csv/Sensitivity/adb_prec_10cv_f12_t700_100times.csv', header=0)

statis_des = []

statis_des.append(df1['acc_score'].describe())
statis_des.append(df2['prec_score'].describe())

print statis_des

#------------------------ #df_10cv = DataFrame({'tree_range': df['tree_range']})
#---------------------------- df_1 = DataFrame({'tree_range': df['tree_range']})
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------- for i in xrange(12,13):
    #------------------------------------- new_str = ff.str_float(df[np.str(i)])
    #----------------------------------- df_1[np.str(i)] = ff.precision(new_str)
#------------------------------------------------------------------------------ 
#--------------------------------------------------------- print df_1.describe()
#-------------------------- df_1.to_csv('adb_prec_63_100_4000.csv', header=True)


# df_6633= pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/gbt_prec_crazy66_33_100_4000.csv', header=0)
#------------------------------------------------------ print df_6633.describe()
#------------------------------------------------------------------------------ 
#------------------------------------------------------- print df_6633.sort('1')



#pp.roll_mean_std(df, 'tree_range', '12','Number of trees', 10)




#tt.plot_gridsearch(df_acc_rf_10cv, aspect = 3)


