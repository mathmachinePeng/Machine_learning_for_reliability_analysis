'''
Created on 29 Sep 2015

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
import seaborn as sns
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.scorer import make_scorer
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
# process data into scaled training and testing



# cls = ['rf', 'adb', 'bag', 'ext', 'gbt']

def main():
    start = timeit.default_timer()
    
    df =pd.read_csv('/home/peng/new160half.csv', header=0)
 #   df['random_number']=np.random.random(size = 160)
  #  df_sort = df.sort(columns='random_number')
#    df_sort.drop(['random_number'], inplace = True, axis = 1)
#    df_sort.to_csv('new_random_160.csv', header = 0)

    p= Preprocessdata.standardprocess()
# #    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis//Test_1/score_long_2features_rf.csv', header=0)
#------------------------------------------------------------------------------ 
    train, trainlabel, test, testlabel =p.noscale(df,0.9)
#    train, trainlabel = p.noaction(df)
# #    df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/score50-500_2.csv', header=0)
    
    ff=RFclass.training()
    tt = RFclass.test()
    feature_range=np.arange(12,13,1)
    tree_range = np.arange(700,701,1)
    
    
    
 
#####################sensitivity for 10cv##############    
    #---------------------------------------------------------------- score = []
    #------------------------------------------------------ for i in range(100):
        # score.append(ff.trainman_sensitivity_CV('adb', train, trainlabel, tree_range, feature_range))
#-------------------------------------------------------------- #    print score
    #- df_raw_times = pd.DataFrame({'times':np.arange(1,101,1), 'scores':score})
#------------------------------------------------------------------------------ 
    #----------------------- df_raw_times = ff.str_float(df_raw_times['scores'])
#------------------------------------------------------------------------------ 
    #---------------------------------- df_acc_times = ff.accuracy(df_raw_times)
    #-- df_acc_times.to_csv('adb_acc_10cv_f12_t700_100times.csv', header = True)



    """Just separate"""
    #===========================================================================
    # forest= ff.trainforest('ext', train, trainlabel,1900,9)
    # y_pred = forest.predict(test)
    # print metrics.precision_score(testlabel,y_pred)
    # cm = metrics.confusion_matrix(testlabel, y_pred)
    # tt.plot_confusion_matrix(cm)
    #===========================================================================
    
    
    """the CART single tree"""
    
    forest = ff.trainforest('cart', train, trainlabel,20,1)
    y_pred = forest.predict(test)
    print metrics.accuracy_score(testlabel,y_pred)
    print metrics.precision_score(testlabel,y_pred)
    cm = metrics.confusion_matrix(testlabel, y_pred)
    tt.plot_confusion_matrix(cm)
    
    

    #---------------------------------------------------------------- score = []
    #------------------------------------------------------ for i in range(100):
        #----------- forest = ff.trainforest('adb', train, trainlabel, 1450, 11)
        #----------------------------------------- y_pred = forest.predict(test)
        #--------------- score.append(metrics.accuracy_score(testlabel, y_pred))
#------------------------------------------------------------------------------ 
    #--------- df=pd.DataFrame({'times': np.arange(1,101,1), 'acc_score':score})
    #--------------- df.to_csv('adb_acc_63_f12_t1450_100times.csv', header=True)
    #------------------------------------------------------- print df.describe()
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    # df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis/Test_1/adb_acc_63_f12_t1450_100times.csv', header=0)
    #------------------------------------ plt.plot(df['times'], df['acc_score'])
 #   plt.xticks(np.arange(1,101,1),np.arange(1,101,1))
#    plt.xlabel()
    plt.show()







############################################################################################################################
    
    #------------------------------------- df_66_33 = {'tree_tange': tree_range}
    #---------------------------------------------- df_all = DataFrame(df_66_33)

 #   scores = ff.trainonlyfeat('bag', train, trainlabel, tree_range, feature_range)
#    scores.to_csv('bag_100_4000_10times.csv', header=True)

    
    # data = ff.train_repeat_forest_metrics('bag', train, trainlabel, test, testlabel, tree_range, feature_range, 10)
    #---------------- data.to_csv('nnnnnn_crazy66_33_100_4000.csv', header=True)
#------------------------------------------------------------------------------ 
    # data = ff.train_repeat_forest_metrics('adb', train, trainlabel, test, testlabel, tree_range, feature_range, 10)
    #------------ data.to_csv('nnnnnnnnnn_crazy66_33_100_4000.csv', header=True)
    
    # data = ff.train_repeat_forest_metrics('gbt', train, trainlabel, test, testlabel, tree_range, feature_range, 10)
    #------------------- data.to_csv('gbt_crazy66_33_100_4000.csv', header=True)

   
    #-- data = ff.trainmanCV('rf', train, trainlabel, tree_range, feature_range)
    #-------------------------- data.to_csv('rf_crazy100_4000.csv', header=True)
#------------------------------------------------------------------------------ 
    #- data = ff.trainmanCV('ext', train, trainlabel, tree_range, feature_range)
    #------------------------- data.to_csv('ext_crazy100_4000.csv', header=True)
    
    
    #- data = ff.trainmanCV('bag', train, trainlabel, tree_range, feature_range)
    #------------------------ data.to_csv('bag_crazy100_4000n.csv', header=True)
#------------------------------------------------------------------------------ 
    #- data = ff.trainmanCV('adb', train, trainlabel, tree_range, feature_range)
    #------------------------ data.to_csv('adb_crazy100_4000n.csv', header=True)
    
    #- data = ff.trainmanCV('gbt', train, trainlabel, tree_range, feature_range)
    #------------------------ data.to_csv('gbt_crazy100_4000n.csv', header=True)


#    scores.to_csv('rf_1_5_1_feature4.csv', header=False)
    
#    print scores
    stop = timeit.default_timer()
    print "The running takes %r min" %((stop-start)/60)

    
    
if __name__ == '__main__':
    main()    
    


    
    
    #-- scores = ff.trainCV('ext', train, trainlabel, tree_range, feature_range)
    #---------------------- scores.to_csv('ext_100_4000_new.csv', header = True)
    
    
    
#------------------------------------------------------------------------------ 
    #- scores1 = ff.trainCV('gbt', train, trainlabel, tree_range, feature_range)
    #--------------------- scores1.to_csv('gbt_100_4000_new.csv', header = True)




#    scores=ff.trainonlyfeat('adb', train, trainlabel, tree_range, feature_range)


    #--------------------------------------------------- for i in feature_range:
        #---------------------------------------------------- score_feature = []
        #-------------------------------------------------- for j in tree_range:
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
            # score = ff.train_repeat_forest('adb', train, trainlabel, test, testlabel, j, i, 10)
            #--------------------------------------- score_feature.append(score)
        #----------------------------------------------- df_all[i]=score_feature
    #--------------- df_all.to_csv('adb_66_33_10_100_4000times.csv',header=True)
    
    


    #--------------------------------------------------- for i in feature_range:
        #---------------------------------------------------- score_feature = []
        #-------------------------------------------------- for j in tree_range:
#------------------------------------------------------------------------------ 
            # score1= ff.train_repeat_forest('adb', train, trainlabel, test, testlabel, j, i, 10)
            #-------------------------------------- score_feature.append(score1)
        #----------------------------------------------- df_all[i]=score_feature
    #--------------- df_all.to_csv('adb_66_33_10_100_4000times.csv',header=True)
    
    
    



    #--------------------------------------------------- for i in feature_range:
        #---------------------------------------------------- score_feature = []
        #-------------------------------------------------- for j in tree_range:
#------------------------------------------------------------------------------ 
            # score2 = ff.train_repeat_forest('gbt', train, trainlabel, test, testlabel, j, i, 10)
            #-------------------------------------- score_feature.append(score2)
        #----------------------------------------------- df_all[i]=score_feature
    #-------------- df_all.to_csv('gbt_66_33_10_2000_4000times.csv',header=True)
    
    
    
    
    
    
    
    
    
# 
# 
#     tt=RFclass.test()    
#     forest= ff.trainforest('rf', train, trainlabel, 500,4)
#     impor_rf = ff.importance(forest, 12)
#===============================================================================


    
    #------------------------------------------- feature_range=np.arange(1,12,1)
    #-------------------------------------- tree_range = np.arange(2000,4010,20)
    #--- scores = ff.trainCV('rf', train, trainlabel, tree_range, feature_range)
    #------------------------ scores.to_csv('rf_2000_4000_20.csv', header= True)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #- scores1 = ff.trainCV('ext', train, trainlabel, tree_range, feature_range)
    #---------------------- scores1.to_csv('ext_2000_4000_20.csv', header= True)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #- scores2 = ff.trainCV('gbt', train, trainlabel, tree_range, feature_range)
    #---------------------- scores2.to_csv('gbt_2000_4000_20.csv', header= True)













    
    
    
    # df = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis//Test_1/gbt_66_33_10times.csv', header=0)
#------------------------------------------------------------------------------ 
    #------------------------------------------------------- print df.describe()
#------------------------------------------------------------------------------ 
    #------------------------------------------------------- scores=np.array(df)
    #---------------------------------------------------- scores=scores[:, 2:].T
#------------------------------------------------------------- #    print scores
    #------------------------------------------------------ scores= scores[:,5:]
    #---------------------------------------------------- print np.shape(scores)
#------------------------------------------------------------------------------ 
#--------------------------------------------- #    print np.arange(100,2010,20)
#------------------------------------------------------------------------------ 
    #------------------------------------------------------------- figsize(16,8)
    #----------------------------------------------- fig, ax = plt.subplots(1,1)
    #----------- cax = ax.imshow(scores, interpolation='none', origin='highest',
                    #--------------------------- cmap=plt.cm.coolwarm, aspect=3)
#------------------------------------------------------------------------------ 
    #------------------ plt.grid(b=True, which='x', color='white',linestyle='-')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    # plt.xticks(np.linspace(-0.5,90.5,10), np.arange(200,2010,200), fontsize = 20)
    #----------- plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
    #------------------------------- plt.xlabel('Number of trees',fontsize = 24)
    #--------------------------- plt.ylabel('Number of features', fontsize = 24)
    #---------------------------------------------- ax.yaxis.grid(False,'major')
    #--------------------------------------------- ax.xaxis.grid(False, 'major')
    # cb = fig.colorbar(cax, orientation='horizontal', pad = 0.15, shrink=1, aspect=50)
    #------------------------------------------- cb.ax.tick_params(labelsize=14)
    
    
    
    



#    ax.yaxis.set_major_locator(MultipleLocator(1))


    
#---------- #    ax.get_xaxis().set_major_locator(mpl.ticker.AutoMinorLocator())
    #----------- ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #----------- ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #--------------- ax.grid(b=False, which='major', color='w', linewidth=0.001)
    #--------------- ax.grid(b=False, which='minor', color='w', linewidth=0.001)
    # cb = fig.colorbar(cax, orientation='horizontal', pad = 0.15, shrink=1, aspect=50)
    #------------------------------------------- cb.ax.tick_params(labelsize=14)
  
#    plt.xticks(np.linspace(0,200,10),np.arange(100,2010,40), fontsize=14)
 #   plt.yticks(np.linspace(-0.5,39.5,11),np.arange(-10,11,2), fontsize=14)
  
     
    #===========================================================================
    # ax = sns.heatmap(scores, cmap="coolwarm", )
    # ax.set_aspect(3)
    #===========================================================================
    
    
    
    
    #--------------------------------------------- figg, axx = plt.subplots(1,2)
    #--------- caxx = axx.imshow(scores, interpolation='none', origin='highest',
                    #--------------------------- cmap=plt.cm.coolwarm, aspect=3)
#------------------------------------------------------------------------------ 
    #------------------ plt.grid(b=True, which='x', color='white',linestyle='-')
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    # plt.xticks(np.linspace(-0.5,90.5,10), np.arange(200,2010,200), fontsize = 20)
    #----------- plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
#------------------------------------------------------------------------------ 
    #--------------------------------------------- axx.yaxis.grid(False,'major')
    #-------------------------------------------- axx.xaxis.grid(False, 'major')
    
    
#    plt.show()  
    
    
    
    
    
    
    
    
    
    #===========================================================================
    # param_grid = dict(max_features=feature_range, n_estimators=tree_range) 
    # 
    # grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)
    # grid.fit(train, trainlabel) 
    # print("The best parameters are %s with a score of %0.2f"
    #           % (grid.best_params_, grid.best_score_))     
    # 
    # scores = [x[1] for x in grid.grid_scores_]
    # scores = np.array(scores).reshape(len(tree_range), len(feature_range))        
    # scores = DataFrame(scores)
    # print scores
    #===========================================================================
    
    #-------------------------------------------------------- seed=["rf", "adb"]
#------------------------------------------------------------------------------ 
    #------------------------------------------------------------ for i in seed:
        #------------------------------------------------------------ score = []
        #-------------------------------------- for j in np.arange(10, 100, 10):
#------------------------------------------------------------------------------ 
            #--------------- forest = ff.trainforest(i, train, trainlabel, j, 4)
            #-------------- score.append(tt.testforest(test, testlabel, forest))
#------------------------------------------------------------------------------ 
#------- #        score = score.set_index([range(0, len(np.arange(10,100,10)))])
#---------------------------------------------------------- #        print score
        #--------------------------------------------------------- df_2[i]=score
#------------------------------------------------------------------------------ 
    #------------------------------------- df_2.to_csv('new.csv', header = True)
        

    
    
    
    ####################plot confusion matrix###############
    #------------------------------------------- cm=np.array([[17, 6], [4, 21]])
    #---------------------------------------------- tt.plot_confusion_matrix(cm)

    ##########################rf####################
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






