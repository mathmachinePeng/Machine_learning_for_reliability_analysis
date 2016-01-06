'''
Created on 20 Oct 2015

@author: peng
'''
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from scipy.interpolate import spline
from bcolz.toplevel import arange
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
import timeit
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from IPython.core.pylabtools import figsize
# process data into scaled training and testing



# cls = ['rf', 'adb', 'bag', 'ext', 'gbt']

def main():
    start = timeit.default_timer()
    df =pd.read_csv('/home/peng/new160half.csv', header=0)
#data=standardprocess()
    p= Preprocessdata.standardprocess()

    
#    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/score50-500_2.csv', header=0)   
    train, trainlabel, test, testlabel = p.scaledivd(df, 0.7)
    #----------------------------------------------------- svcc=mysvc.training()
#------------------------------------------------------------------------------ 
    #-------------- best, scores=svcc.svmsigmoid(train, trainlabel, -10, 10,100)
    #------------------------------------------------------------------------- #
    #--------------------------------------------------------- # # Test with SVM
    #--------------------------------------------------------- svtt=mysvc.test()
    #--------------------------------------- svtt.testsvm(test, testlabel, best)
    #df_6s= pd.read_csv('/media/peng/Data/Project/CORROSION/DATA/alloy_600_data.csv', header=0)
  
    
    svcc=mysvc.training()
    best, scores=svcc.svmpoly(train, trainlabel, -10, 10,-10, 10, 40, plot=True)
    print scores


    score_test = DataFrame(scores)
    score_test.to_csv('SVM_score_test_.csv')
    
    
    
    # scores = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/SVMrbf_score_test_101040.csv', header=0)
    #-------------------------------------------------------------- print scores
    #--------------------------------------------------- scores=np.array(scores)
    #------------------------------------------------------ scores=scores[:, 1:]
#------------------------------------------------------------- #    print scores


#   CS = plt.contour(X, Y, scores)

    #------------------------------------------------------------ figsize(8,6.5)
#------------------------------------------------------------------------------ 
    #------------ plt.imshow(scores, interpolation='bilinear', origin='highest',
               #--------------------- cmap=plt.cm.coolwarm, vmin=0.4, vmax=0.85)
   #------------------------------------------------------------ # Test with SVM
    #------------------------------------------------------- cb = plt.colorbar()
#------------------------------------------------------------------------------ 
    #---- plt.xticks(np.linspace(-0.5,39.5,11),np.arange(-10,11,2), fontsize=14)
    #---- plt.yticks(np.linspace(-0.5,39.5,11),np.arange(-10,11,2), fontsize=14)
#------------------------------------------------------------------------------ 
    #------------------------------------------- cb.ax.tick_params(labelsize=14)
    #---------------------------------- plt.xlabel('$log_2\gamma$', fontsize=24)
    #--------------------------------------- plt.ylabel('$log_2C$', fontsize=24)
    #---------------------------------------------------------------- plt.show()
    
    

    stop = timeit.default_timer()
    print "The running takes %r s" %(stop-start)    
if __name__ == '__main__':
    main()    
 

"""generate validation set"""
#-------------------------------------------------------------- len = len(train)
#------------------------------------------------------ valid = train[80:len, :]
#----------------------------------------------- validlabel = trainlabel[80:len]
#----------------------------------------------------------- train=train[0:80,:]
#------------------------------------------------- trainlabel = trainlabel[0:80]
#------------------------------------------------------------------------------ 
#--------- dataset=[(train, trainlabel), (valid, validlabel), (test, testlabel)]





""" This is SVM"""
# Train with SVM
#--------------------------------------------------------- svcc=MySVM.training()
#-------------------- best, scores=svcc.svmlinear(train, trainlabel, -10, 10,41)
#----------------------------------------------------------------------------- #
#------------------------------------------------------------- # # Test with SVM
#------------------------------------------------------------- svtt=mysvc.test()
#------------------------------------------- svtt.testsvm(test, testlabel, best)
 
"""This is SVM"""
#===============================================================================
# # # Train with SVM
# svcc=mysvc.training()
# best, scores=svcc.svmlinear(train, trainlabel, -10, 10,20)
# # 
# # # Test with SVM
# svtt=mysvc.test()
# svtt.testsvm(test, testlabel, best)
#===============================================================================
 
 
 
# Train with SVM
#===============================================================================
# svcc=mysvc.training()
# best, scores=svcc.svmrbf(train, trainlabel, -10, 10,-10, 10, 41, )
#  
# # Test with SVM
# svtt=mysvc.test()
# svtt.testsvm(test, testlabel, best)
#===============================================================================

#===============================================================================
# n_group = 5
# linearvalue=(76, 81.25, 86, 76, 81)
# polyvalue= (76, 73, 75, 72, 73)
# rbfvalue=(79, 83, 79, 92, 85)
# 
# fig, ax= plt.subplots()
# 
# index=np.arange(n_group)
# bar_width= 0.35
# 
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
# 
# rects1 = plt.bar(index, linearvalue, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  
#                  error_kw=error_config,
#                  label='linear')
# 
# rects2 = plt.bar(index + bar_width, rbfvalue, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  
#                  error_kw=error_config,
#                  label='polynomial')
# 
# rects2 = plt.bar(index + 2*bar_width, rbfvalue, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  
#                  error_kw=error_config,
#                  label='rbf')
# 
# plt.xlabel('Classification metrics')
# plt.ylabel('Percentage value (%)')
# #plt.title('Scores by group and gender')
# 
# ax.set_xticks(index+5*bar_width)
# plt.xticks(index + 1.5*bar_width, ('CV accuracy', 'Test accuracy', 'Precision', 'Recall', 'F1 score'))
# plt.legend()
# 
# plt.tight_layout()
# plt.show()
#===============================================================================