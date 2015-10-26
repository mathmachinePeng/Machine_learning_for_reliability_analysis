'''
Created on 29 Sep 2015

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
from bcolz.toplevel import arange
import timeit

# process data into scaled training and testing



# cls = ['rf', 'adb', 'bag', 'ext', 'gbt']

def main():
    start = timeit.default_timer()
    df =pd.read_csv('/home/peng/new160half.csv', header=0)
#data=standardprocess()
    p= Preprocessdata.standardprocess()
#    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis//Test_1/score_long_2features_rf.csv', header=0)
    xx= np.arange(10, 100, 10)
    
    df_2 = pd.DataFrame(xx)
    train, trainlabel, test, testlabel = p.noscale(df, 0.7)  
#    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/score50-500_2.csv', header=0)   
    ff=RFclass.training()


    tt=RFclass.test()    
    
    seed=["rf", "adb"]

    for i in seed:
        score = []
        for j in np.arange(10, 100, 10):
            
            forest = ff.trainforest(i, train, trainlabel, j, 4)
            score.append(tt.testforest(test, testlabel, forest))
   
#        score = score.set_index([range(0, len(np.arange(10,100,10)))])
#        print score
        df_2[i]=score
            
    df_2.to_csv('new.csv', header = True)                  
        

    
    
    
    ######################plot confusion matrix###############
    #------------------------------------------- cm=np.array([[17, 6], [4, 21]])
    #---------------------------------------------- tt.plot_confusion_matrix(cm)
       
    ###########################rf####################
    #------------------------------------------------------- score_input_rf = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #---------------- forest = ff.trainforest('rf', train, trainlabel, i, 2)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #--------------------------------------------- score_input_rf.append(LL)
#----------------------------------------------------------------- #       d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #------------------------------------------------ df_2['rf2']=score_input_rf
    #------------------- df_2.to_csv('score_long_2features_rf.csv', header=True)
    
    #------------------------ ###########################ext####################
    #------------------------------------------------------ score_input_ext = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('ext', train, trainlabel, i, 2)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_ext.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #---------------------------------------------- df_2['ext2']=score_input_ext
    #------------------ df_2.to_csv('score_long_2features_ext.csv', header=True)
    #------------------------------------------------------- print "finshed one"
        #-------------------- ###########################gbt####################
    #------------------------------------------------------ score_input_gbt = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('gbt', train, trainlabel, i, 2)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_gbt.append(LL)
 #---------------------------------------------------------------- #       d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #---------------------------------------------- df_2['gbt2']=score_input_gbt
    #------------------ df_2.to_csv('score_long_2features_gbt.csv', header=True)
    #------------------------------------------------------- print "finshed one"
# ##########################finish one circle ########################################
    #------------------------- ###########################rf####################
    #------------------------------------------------------- score_input_rf = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #---------------- forest = ff.trainforest('rf', train, trainlabel, i, 4)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #--------------------------------------------- score_input_rf.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #------------------------------------------------ df_2['rf4']=score_input_rf
    #------------------- df_2.to_csv('score_long_4features_rf.csv', header=True)
    #------------------------------------------------------- print "finshed one"
    #------------------------ ###########################ext####################
    #------------------------------------------------------ score_input_ext = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('ext', train, trainlabel, i, 4)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_ext.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #---------------------------------------------- df_2['ext4']=score_input_ext
    #------------------ df_2.to_csv('score_long_4features_ext.csv', header=True)
    #------------------------------------------------------- print "finshed one"
        #-------------------- ###########################gbt####################
    #------------------------------------------------------ score_input_gbt = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('gbt', train, trainlabel, i, 4)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_gbt.append(LL)
 #---------------------------------------------------------------- #       d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #---------------------------------------------- df_2['gbt4']=score_input_gbt
    #------------------ df_2.to_csv('score_long_4features_gbt.csv', header=True)
# ##########################finish one circle ########################################
#------------------------------------------------------------------------------ 
    #------------------------- ###########################rf####################
    #------------------------------------------------------- score_input_rf = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #---------------- forest = ff.trainforest('rf', train, trainlabel, i, 6)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #--------------------------------------------- score_input_rf.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #------------------------------------------------ df_2['rf6']=score_input_rf
    #------------------- df_2.to_csv('score_long_6features_rf.csv', header=True)
#------------------------------------------------------------------------------ 
    #------------------------ ###########################ext####################
    #------------------------------------------------------ score_input_ext = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('ext', train, trainlabel, i, 6)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_ext.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #---------------------------------------------- df_2['ext6']=score_input_ext
    #------------------ df_2.to_csv('score_long_6features_ext.csv', header=True)
    #------------------------------------------------------- print "finshed one"
        #-------------------- ###########################gbt####################
    #------------------------------------------------------ score_input_gbt = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('gbt', train, trainlabel, i, 6)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_gbt.append(LL)
 #---------------------------------------------------------------- #       d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #---------------------------------------------- df_2['gbt6']=score_input_gbt
    #------------------ df_2.to_csv('score_long_6features_gbt.csv', header=True)
# ##########################finish one circle ########################################
    #------------------------- ###########################rf####################
    #------------------------------------------------------- score_input_rf = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #--------------- forest = ff.trainforest('rf', train, trainlabel, i, 10)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #--------------------------------------------- score_input_rf.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #----------------------------------------------- df_2['rf10']=score_input_rf
    #------------------ df_2.to_csv('score_long_10features_rf.csv', header=True)
#------------------------------------------------------------------------------ 
    #------------------------ ###########################ext####################
    #------------------------------------------------------ score_input_ext = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #-------------- forest = ff.trainforest('ext', train, trainlabel, i, 10)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_ext.append(LL)
#---------------------------------------------------------------- #        d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #--------------------------------------------- df_2['ext10']=score_input_ext
    #----------------- df_2.to_csv('score_long_10features_ext.csv', header=True)
#------------------------------------------------------------------------------ 
        #-------------------- ###########################gbt####################
    #------------------------------------------------------ score_input_gbt = []
#-------------------------------------------------------------------- #    d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #-------------- forest = ff.trainforest('gbt', train, trainlabel, i, 10)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #-------------------------------------------- score_input_gbt.append(LL)
 #---------------------------------------------------------------- #       d=d+1
# #        print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
    #------------------------------------------------------- print "finshed one"
    #--------------------------------------------- df_2['gbt10']=score_input_gbt
    #----------------- df_2.to_csv('score_long_10features_gbt.csv', header=True)
##########################finish one circle ########################################       
    #------------------------------------------------------- score_input_rf = []
    #--------------------------------------------------------------------- d = 0
    #----------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
        #------------------ forest = ff.trainforest('bag', train, trainlabel, i)
        #---------------------------- LL= tt.testforest(test, testlabel, forest)
        #--------------------------------------------- score_input_rf.append(LL)
        #----------------------------------------------------------------- d=d+1
        # print "Now we finish  %d epoch, there are %r in all" %(d, len(spanx)-d)
#------------------------------------------------------------------------------ 
    #------------------------------------------------- df_2['rf']=score_input_rf
    
    
    
    

    
    
    
    
#   plt.plot(spanx, score_input_bag)
    stop = timeit.default_timer()
    print "The running takes %r s" %(stop-start)

    
    
if __name__ == '__main__':
    main()    
    
    
    
    
""" THis is RF"""
# Train the model by RF


#--------------------------------------------------------- ff=RFclass.training()
#------------------------- #forest= ff.trainforest('rf', train, trainlabel, 500)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------- tt=RFclass.test()
#------------------------------------ #LL=tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
#------------------------------------------------ spanx= np.arange(10, 1000, 10)
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #----------------------- forest = ff.trainforest('rf', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#-------------------------------------------------- plt.plot(spanx, score_input)
#-------------------------------------------------------------------- plt.show()

#relative importances of features forest"""


    #---------------------------------------------------- ff= RFclass.training()
#------------------------------------------------------------------------------ 
    #---------------------- forest= ff.trainforest('rf', train, trainlabel, 500)
    #-------------------------------------- impor_rf = ff.importance(forest, 12)
#------------------------------------------------------------------------------ 
    #----------------------------------------------------- x=np.arange(1, 13, 1)
    #------------------------------------------------------ impor = DataFrame(x)
    #------------------------------------------------------ impor['rf']=impor_rf
#------------------------------------------------------------------------------ 
    #--------------------- forest= ff.trainforest('adb', train, trainlabel, 500)
    #------------------------------------- impor_adb = ff.importance(forest, 12)
#------------------------------------------------------------------------------ 
    #----------------- forest= ff.trainforest('bagging', train, trainlabel, 500)
    #------------------------------------- impor_bag = ff.importance(forest, 12)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #--------------------- forest= ff.trainforest('ext', train, trainlabel, 500)
    #------------------------------------- impor_ext = ff.importance(forest, 12)
#------------------------------------------------------------------------------ 
    #--------------------- forest= ff.trainforest('gbt', train, trainlabel, 500)
    #------------------------------------- impor_gbt = ff.importance(forest, 12)
#------------------------------------------------------------------------------ 
    #---------------------------------------------------- impor['adb']=impor_adb
    #----------------------------------------------- impor[ 'bagging']=impor_bag
    #--------------------------------------------------- impor[ 'ext']=impor_ext
    #-------------------------------------------------- impor[ 'gbt']= impor_gbt
    #-------------------------- impor.to_csv('relativeimpor.csv', header = True)
#------------------------------------------------------------------------------ 
    #--------------------------------------------------------------- print impor

#find the best number of trees
#-------------------------------------- cls = ['rf', 'adb', 'bag', 'ext', 'gbt']
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------- spanx= np.arange(1, 510, 10)
#----------------------------------------------------- output = DataFrame(spanx)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------- ###########################rf
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #----------------------- forest = ff.trainforest('rf', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#------------------------------------------------------ output['rf']=score_input
#----------------------------------------------- ############################adb
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #---------------------- forest = ff.trainforest('adb', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#----------------------------------------------------- output['adb']=score_input
#------------------------------------------------------------------------------ 
#----------------------------------------------------------------- #print output
#------------------------------------------------------------------------------ 
#----------------------------------------------- ############################bag
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #---------------------- forest = ff.trainforest('bag', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#----------------------------------------------------- output['bag']=score_input
#------------------------------------------------------------------------------ 
#----------------------------------------------------------------- #print output
#------------------------------------------------------------------------------ 
#----------------------------------------------- ############################ext
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #---------------------- forest = ff.trainforest('ext', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#----------------------------------------------------- output['ext']=score_input
#------------------------------------------------------------------------------ 
#----------------------------------------------------------------- #print output
#------------------------------------------------------------------------------ 
#--------------------------- ################################################gbt
#------------------------------------------------------------------------------ 
#-------------------------------------------------------------- score_input = []
#------------------------------------------------------------------------------ 
#--------------------------------------------------------------- for i in spanx:
#------------------------------------------------------------------------------ 
    #---------------------- forest = ff.trainforest('gbt', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #---------------------------------------------------- score_input.append(LL)
#------------------------------------------------------------------------------ 
#----------------------------------------------------- output['gbt']=score_input
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------ print output
#------------------------------------------------------------------------------ 
#------------------------------- output.to_csv('score50-500_8.csv', header=True)
#------------------------------------------------------------------ print output


#############################Plot the number of estimator

#----------------------------------------------------------- n_bagging_score =[]
#-------------------------------------------------- spanx= np.arange(1, 510, 10)
#--------------------------------------------------------------- for i in spanx:
    #---------------------- forest = ff.trainforest('adb', train, trainlabel, i)
    #-------------------------------- LL= tt.testforest(test, testlabel, forest)
    #------------------------------------------------ n_bagging_score.append(LL)
#------------------------------------------------------------------------------ 
#--------------------------------- outperfor={'n_estimator':spanx,'rf_score':LL}
#------------------------------------------------- outframe=DataFrame(outperfor)
#---------------------------------------------------------------- print outframe
#----------------------------------------------------- print outframe.describe()
#---------------------------- x_new = np.linspace(spanx.min(), spanx.max(), 100)
#---------------------------- power_line = spline(spanx, n_bagging_score, x_new)
#---------------------- #plt.plot(x_new, power_line, linewidth = 2, color = 'r')
#------------------------------- plt.plot(spanx, n_bagging_score, linewidth = 2)
#--------------------------------------------- #plt.xticks(spanx ,fontsize = 12)
#----------------------------------------------- plt.xticks(np.arange(0,501,50))
#------------------------------------------------------------ plt.xlim([0, 500])
#-------------------------------------------------------------- plt.ylim([0, 1])
#-------------------------------------------------------------------- plt.show()






#dependences of features 
#feature_set = (8,9)
#feature_names=["feature 0", "1","2","3","4",""""""""""""""""""""""""]
#ff.dependence3d(forest, train, feature_set)


"""calculate the output one by one"""

    #------------------ forest = ff.trainforest('bag', train, trainlabel, 10000)
    #------------------------------- bag= tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
    #------------------ forest = ff.trainforest('adb', train, trainlabel, 10000)
    #------------------------------ adb = tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
    #------------------- forest = ff.trainforest('rf', train, trainlabel, 10000)
    #------------------------------- rf = tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
    #------------------ forest = ff.trainforest('ext', train, trainlabel, 10000)
    #------------------------------ ext = tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
    #------------------ forest = ff.trainforest('gbt', train, trainlabel, 10000)
    #------------------------------ gbt = tt.testforest(test, testlabel, forest)
#------------------------------------------------------------------------------ 
    #----------- test_bag= {'bag':bag, 'adb':adb, 'rf':rf, 'ext':ext, 'gbt':gbt}
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #---------------------------------------------- test_bag=DataFrame(test_bag)
#------------------------------------------------------------------------------ 
#----------------------------------------------------------- #    print test_bag
#------------------------------------------------------------------------------ 
    #-------------------------- test_bag.to_csv('test_bag_try.csv', header=True)