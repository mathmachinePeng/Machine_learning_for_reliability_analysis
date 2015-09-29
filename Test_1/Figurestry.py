'''
Created on 28 Sep 2015

@author: peng
'''
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



N = 4

bag = (0.79, 0.80, 0.80, 0.80)
adb = (0.73, 0.77, 0.68, 0.72)
rf = (0.81, 0.81, 0.84, 0.82)
ext = (0.79, 0.78, 0.84, 0.81)
gbt = (0.79, 0.80, 0.80, 0.80)

all_score = {'00bag':bag, '01adb':adb, '02rf':rf, '03ext':ext, '04gbt':gbt}

all_score = DataFrame(all_score)

all_score.to_csv('all_scores.csv', header = 0)
#------------------------ ind = np.arange(N)    # the x locations for the groups
#------- width = 0.35       # the width of the bars: can also be len(x) sequence
#------------------------------------------------------------------------------ 
#------------------- p1 = plt.bar(ind, bag,   width, color='r', label='Bagging')
#----------------- p2 = plt.bar(ind, adb, width, color='y', label='Adaboosting',
             #------------------------------------------------------ bottom=bag)
#------------------------- p3 = plt.bar(ind, rf, width, color='b', label = 'RF',
             #------------------------------------------------------ bottom=adb)
#----------------------- p4 = plt.bar(ind, ext, width, color='w', label = 'ERT',
             #------------------------------------------------------- bottom=rf)
#------------------------- p5 = plt.bar(ind, gbt, width, color='g', label='GTB',
             #------------------------------------------------------ bottom=ext)
#------------------------------------------------------------------------------ 
#---------------------------------------------------------- plt.ylabel('Scores')
#--------------------------------------- plt.title('Scores by group and gender')
#--------------------------- plt.xticks(ind+width/2., ('G1', 'G2', 'G3', 'G4') )
#----------------------------------------------- #plt.yticks(np.arange(0,81,10))
#------------------------------------------------------------------ plt.legend()
#------------------------------- #plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------------- plt.show()

#------------------------------------------------------------------------------ 
#------------------------------------------------- spanx = np.arange(1, 510, 10)
#------------------------------------------------------------------------------ 
#---------------------------- x_new = np.linspace(spanx.min(), spanx.max(), 200)
#---------------------------- power_line_adb = spline(spanx, df_6['adb'], x_new)
#-------------------------- power_line_bag = spline(spanx, df_bag['bag'], x_new)
#--------------------------- #power_line_gbt = spline(spanx, df_6['gbt'], x_new)
#------------------------------------------------------------------------------ 
# #plt.plot(x_new, power_line_bag, linewidth = 2, label='ERT', color ='#00aa00')
# plt.plot(x_new, power_line_bag, linewidth = 2, label ='Bagging', color='#ff5500')
# plt.plot(x_new, power_line_adb, linewidth = 2, label='Adaboosting', color='#00aaff')
#------------------------------------------------------------------------------ 
# #------------------------------------ plt.plot(spanx, df_2['rf'], linewidth = 2)
# #----------------------------------- plt.plot(spanx, df_2['ext'], linewidth = 2)
# #----------------------------------- plt.plot(spanx, df_2['gbt'], linewidth = 2)
#--------------------------------------------- #plt.xticks(spanx ,fontsize = 12)
#---------------------------------------------- #plt.xticks(np.arange(0,501,50))
#------------------------------------ plt.xlabel('Number of trees', fontsize=16)
#------------------------------------------- plt.ylabel('Accuracy', fontsize=16)
#---------------------------------------- plt.legend(fontsize=16, frameon=False)
#----------------------------------------------------------- plt.xlim([50, 500])
#--------------------------------------------------------- plt.ylim([0.6, 0.95])
#-------------------------------------------------------------------- plt.show()



