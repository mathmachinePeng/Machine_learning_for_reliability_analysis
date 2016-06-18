from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.scorer import make_scorer
import re
from IPython.core.pylabtools import figsize
import seaborn

class fancy():    
    def __init__(self):
        print "*******************************************"    
    
    def plot_confusion_matrix(self, cm):
        norm_conf = []
        for i in cm:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)
        
        fig = plt.figure(figsize=(8,6.5))
        
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, 
                        interpolation='nearest')
        
        width = len(cm)
        height = len(cm[0])
        
        #----------------------------------------------- for x in xrange(width):
            #------------------------------------------ for y in xrange(height):
                #------------------------- ax.annotate(str(cm[x][y]), xy=(y, x),
                            #--------------------- horizontalalignment='center',
                            # verticalalignment='center', fontsize=40, color='#00aa00', fontweight='bold')
        for x in xrange(width):
            for y in xrange(height):
                if x ==y:
                    
                    ax.annotate(str(cm[x][x]), xy=(x, x), 
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=40, color='#FFFFFF', fontweight='bold')
                else:
                    

                    ax.annotate(str(cm[y][x]), xy=(x, y),
                            horizontalalignment='center',
                             verticalalignment='center', fontsize=40, color='#00aaff', fontweight='bold')
        
        cb = fig.colorbar(res)
        cb.ax.tick_params(labelsize=14)

    
        plt.xticks(np.arange(2), ['Unfailed','Failed'], fontsize=20)
        plt.yticks(np.arange(2), ['Unfailed','Failed'], fontsize=20, rotation=90)
        plt.xlabel('True class', fontsize=24)
        plt.ylabel('Predicted class', fontsize = 24)
        plt.show()
    
    def plot_gridsearch(self,x, aspect = 3):

        scores=np.array(x)
        scores=scores[:, 1:].T
        #    print scores
        #scores= scores[:,5:]
        print np.shape(scores)
        
        #    print np.arange(100,2010,20)
        
        figsize(16,8)
        fig, ax = plt.subplots(1,1)
        cax = ax.imshow(scores, interpolation='none', origin='highest',
                        cmap=plt.cm.coolwarm, aspect=aspect)
        
        plt.grid(b=True, which='x', color='white',linestyle='-')
        
        plt.xlim(0,96)
        
        plt.xticks((-0.5, 96.5), (100,2000), fontsize = 20)        
#        plt.xticks(np.linspace(0,194,10), int(np.linspace(100,4000,10)), fontsize = 20)
        plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
        plt.xlabel('Number of trees',fontsize = 24)
        plt.ylabel('Number of features', fontsize = 24)
        ax.yaxis.grid(False,'major')
        ax.xaxis.grid(False, 'major')
        cb = fig.colorbar(cax, orientation='horizontal', pad = 0.15, shrink=1, aspect=50)
        cb.ax.tick_params(labelsize=14)
        #----------------------------------------------------------- scores=np.array(df)
        #------------------------------------------------------ scores=scores[:, 2:13].T
        #-------------------- plt.imshow(scores, interpolation='None', origin='highest',
                      #--------------------------------- cmap=plt.cm.coolwarm, aspect=3)
        #------------------------------------------------------------------------------ 
        #-------------- cb = plt.colorbar(orientation='horizontal', shrink=1, aspect=50)
        #---------------------------------------------- #sns.axes_style(axes.grid=False)
        #-------------------------------------------------------------------- plt.show()
        plt.show()

    def roll_mean_std(self, df, xaxis, yaxis, xlabel_name, width):
        figsize(13.5,8)
        
        df = df
        
        fig, ax1 = plt.subplots()


        ax1.plot(df[xaxis][0:96], pd.rolling_mean(df[yaxis][0:96], window = width), 'b-')
        ax1.set_xlabel(xlabel_name,fontsize = 24)
        plt.xticks(fontsize = 20)
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Rolling mean', color='b', fontsize = 24)
        plt.yticks(fontsize = 20)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')


        ax2 = ax1.twinx()
        ax2.plot(df[xaxis][0:96], pd.rolling_std(df[yaxis][0:96], window = width), 'r-')
        ax2.set_ylabel('Rolling standard deviation', color='r',fontsize = 24)
        plt.yticks(fontsize = 20)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
            
        #- ax1.plot(df[xaxis], pd.rolling_mean(df[yaxis], window = width), 'b-')
        #----------------------------- ax1.set_xlabel(xlabel_name,fontsize = 24)
        #--------------------------------------------- plt.xticks(fontsize = 20)
        #--------- # Make the y-axis label and tick labels match the line color.
        #-------------- ax1.set_ylabel('Rolling mean', color='b', fontsize = 24)
        #--------------------------------------------- plt.yticks(fontsize = 20)
        #-------------------------------------- for tl in ax1.get_yticklabels():
            #------------------------------------------------- tl.set_color('b')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
        #----------------------------------------------------- ax2 = ax1.twinx()
        #-- ax2.plot(df[xaxis], pd.rolling_std(df[yaxis], window = width), 'r-')
        #- ax2.set_ylabel('Rolling standard deviation', color='r',fontsize = 24)
        #--------------------------------------------- plt.yticks(fontsize = 20)
        #-------------------------------------- for tl in ax2.get_yticklabels():
            #------------------------------------------------- tl.set_color('r')
#------------------------------------------------------- #        plt.xlim(0,96)
        
        plt.xticks(fontsize = 20)  
        ax1.yaxis.grid(False,'minor')
        ax1.xaxis.grid(False, 'minor')
        ax2.yaxis.grid(False,'major')
        ax2.xaxis.grid(False, 'major')        
        plt.show()
