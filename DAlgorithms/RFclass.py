"""This is the class containing ensemble methods like random forestg, bagging and boosting 
which contains a lot single learners and output the final result based on the voting of every
single learner

Peng Jiang 
27/09/2015 Happy middle moon festival"""


from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import csv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, r2_score, mean_squared_error
from sklearn.metrics.scorer import make_scorer
import re
from IPython.core.pylabtools import figsize
from sklearn.tree import DecisionTreeClassifier


class training(object):
    def __init__(self):
        print "This is conducted in the off-line training phase: \n"
        
    def trainforest(self, seed, train, trainlabel, number_trees, number_features):
        seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                      'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                      'bagging': RandomForestClassifier(n_estimators= number_trees, max_features=12),
                      'cart': DecisionTreeClassifier(criterion='entropy'),
                      'rf_regress': RandomForestRegressor(n_estimators= number_trees, max_features=number_features)}
        rawforest=seed_of_tree[seed]
        forest=rawforest.fit(train,trainlabel)
        outputtrain= forest.predict(train)
        print r2_score(trainlabel, outputtrain) 
        print mean_squared_error(trainlabel, outputtrain)
#        accuracytrain = accuracy_score(trainlabel, outputtrain)        
#        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1])
        #---------------------------------------- print "The method is %r" %seed
        # print "The accuracy for the training set is %r" %accuracytrain, "and the confusion matrix is"
        #------------------------ print confusion_matrix(outputtrain,trainlabel)
        return (forest)
    
    def trainforest_regress(self, seed, train, trainlabel, number_trees, number_features):
        seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                      'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                      'bagging': RandomForestClassifier(n_estimators= number_trees, max_features=12),
                      'cart': DecisionTreeClassifier(criterion='entropy'),
                      'rf_regress': RandomForestRegressor(n_estimators= number_trees, max_features=number_features)}
        rawforest=seed_of_tree[seed]
        forest=rawforest.fit(train,trainlabel)
        outputtrain= forest.predict(train)
        print r2_score(trainlabel, outputtrain) 
#        accuracytrain = accuracy_score(trainlabel, outputtrain)        
#        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1])
        #---------------------------------------- print "The method is %r" %seed
        # print "The accuracy for the training set is %r" %accuracytrain, "and the confusion matrix is"
        #------------------------ print confusion_matrix(outputtrain,trainlabel)
        return (forest)
    
    
    def train_repeat_forest(self, seed, train, trainlabel, test, testlabel, number_trees, number_features, repeat_times):
        seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                      'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                      'bagging': RandomForestClassifier(n_estimators= number_trees, max_features=12)}
        rawforest=seed_of_tree[seed]
        score_list=[]
        for i in np.arange(repeat_times):
            forest=rawforest.fit(train,trainlabel)
            outputtest= forest.predict(test) 
            accuracytrain = accuracy_score(testlabel, outputtest)
            score_list.append(accuracytrain)
        score = np.mean(score_list)
        return score
            
    def train_repeat_forest_metrics(self, seed, train, trainlabel, test, testlabel, tree_range, feature_range, repeat_times):
        datametrics = DataFrame({'tree_range': tree_range})
        
        for number_features in feature_range:
            score1=[]
            for number_trees in tree_range:
                
                seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                  'adb': AdaBoostClassifier(n_estimators= number_trees),
                  'bag': BaggingClassifier(n_estimators= number_trees),
                  'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                  'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                  'bagging': RandomForestClassifier()}
                   
                rawforest = seed_of_tree[seed]
                cm = np.zeros(len(np.unique(testlabel)) ** 2)
                for times in np.arange(repeat_times):
                    forest=rawforest.fit(train,trainlabel)
                    out_value = forest.predict(test)
                    cm += metrics.confusion_matrix(testlabel, out_value).flatten()
                cm_ava = cm/ repeat_times
                score1.append(cm_ava)
                
                
                     
                
                
                                
#                score1.append(training.mean_scores(self, train, trainlabel, rawforest, repeat_times))
        
                
            datametrics[number_features]=score1
        return datametrics
               
        
        

    
#        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1])
        #---------------------------------------- print "The method is %r" %seed
        # print "The accuracy for the training set is %r" %accuracytrain, "and the confusion matrix is"
        #------------------------ print confusion_matrix(outputtrain,trainlabel)
  
    
    
    #-- def manCV(self, seed, train, trainlabel, tree_range, feature_range, cv):
        #----------------------- seed_of_tree = {'rf': RandomForestClassifier(),
                      #---------------------------- 'adb': AdaBoostClassifier(),
                      #----------------------------- 'bag': BaggingClassifier(),
                      #-------------------------- 'ext': ExtraTreesClassifier(),
                      #-------------------- 'gbt': GradientBoostingClassifier(),
                      #-------------------- 'bagging': RandomForestClassifier()}
        #---------------------------------------- rawforest = seed_of_tree[seed]
        #-------------- k_fold = k_fold = cross_validation.KFold(len(train), cv)
#------------------------------------------------------------------------------ 
        #---------------------- for k, (train, trainlabel) in enumerate(k_fold):
            
    def compute_measures(self, tn, fp, fn, tp):
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
        return sensitivity, specificity, fmeasure   
    
         
    def mean_scores(self, X, y, clf, skf):

        cm = np.zeros(len(np.unique(y)) ** 2)
        for i, (train, test) in enumerate(skf):
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            cm += metrics.confusion_matrix(y[test], y_pred).flatten()
#            print "cm is", cm 
        cm_ava = cm/ skf.n_folds
        return np.array(cm_ava)
#        return training.compute_measures(self,*cm / skf.n_folds)
            



#finaly find the 10-fold crossvalidation
    def trainmanCV(self, seed, train, trainlabel, tree_range, feature_range):

         
        
        k_fold = cross_validation.KFold(len(train), n_folds = 10)
        datametrics = DataFrame({'tree_range': tree_range})
        
        for number_features in feature_range:
            score1=[]
            for number_trees in tree_range:
                
                seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                  'adb': AdaBoostClassifier(n_estimators= number_trees),
                  'bag': BaggingClassifier(n_estimators= number_trees),
                  'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                  'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                  'bagging': RandomForestClassifier()}
                   
                rawforest = seed_of_tree[seed]
                                
                score1.append(training.mean_scores(self, train, trainlabel, rawforest, k_fold))
        
                
            datametrics[number_features]=score1
        return datametrics

    def trainman_sensitivity_CV(self, seed, train, trainlabel, tree_range, feature_range):
# this is for sensitivity tests for 10cv only!
         
        
        k_fold = cross_validation.KFold(len(train), n_folds = 10)
        datametrics = DataFrame({'tree_range': tree_range})
        
        for number_features in feature_range:
        #    score1=[]
            for number_trees in tree_range:
                
                seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                  'adb': AdaBoostClassifier(n_estimators= number_trees),
                  'bag': BaggingClassifier(n_estimators= number_trees),
                  'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                  'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                  'bagging': RandomForestClassifier()}
                   
                rawforest = seed_of_tree[seed]
                                
            return training.mean_scores(self, train, trainlabel, rawforest, k_fold)
        
                
           # datametrics[number_features]=score1
        #return datametrics

    
    
    
    def trainCV(self, seed, train, trainlabel, tree_range, feature_range):
        seed_of_tree = {'rf': RandomForestClassifier(), 
                      'adb': AdaBoostClassifier(),
                      'bag': BaggingClassifier(),
                      'ext': ExtraTreesClassifier(),
                      'gbt': GradientBoostingClassifier(),
                      'bagging': RandomForestClassifier()}
         
        rawforest = seed_of_tree[seed]
        param_grid = dict(max_features=feature_range, n_estimators=tree_range)
        
        
        

        grid = GridSearchCV(rawforest, param_grid=param_grid, cv=10, n_jobs=-1)
        grid.fit(train, trainlabel)
        print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))

        acc_scores = [x[1] for x in grid.grid_scores_]
        acc_scores = np.array(acc_scores).reshape(len(tree_range), len(feature_range))
        scores = DataFrame(acc_scores)


        return scores
    
    

        
        
        
    
    
    
    def trainonlyfeat(self, seed, train, trainlabel, tree_range, feature_range):
        seed_of_tree = {'rf': RandomForestClassifier(), 
                      'adb': AdaBoostClassifier(),
                      'bag': BaggingClassifier(),
                      'ext': ExtraTreesClassifier(),
                      'gbt': GradientBoostingClassifier(),
                      'bagging': RandomForestClassifier()}
         
        rawforest = seed_of_tree[seed]
        param_grid = dict( n_estimators=tree_range)
        grid = GridSearchCV(rawforest, param_grid=param_grid, cv=10)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))     
        
        scores = [x[1] for x in grid.grid_scores_]
#        scores = np.array(scores).reshape(len(feature_range))        
        scores = DataFrame(scores)
        return scores
    
    
    
    def importance(self, forest, n, color, plot_std = False):
        figsize(10,8)
        
        print "************************this is the output of relative importance**************"
        #print(forest.feature_importances_)
        importances=forest.feature_importances_
        #return importances
        #std=np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices=np.argsort(importances)[::-1]
        #print indices
        print("Feature ranking:")
        for f in range(12):
              print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))
        #plt.figure(figsize=(8,6.5))
        
        

        
        if plot_std == True:
            pass                                 
            #plt.bar(range(12), importances[indices],
            #color=color, yerr=std[indices], align="center")
        
        else:
            plt.bar(range(12), importances[indices],
            color=color, align="center")
        
        plt.xticks(range(n), indices+1, fontsize=20)
        plt.yticks(fontsize = 20)
        plt.xlim([-1, n])
#        plt.ylim([0.00,0.30])
        plt.xlabel('The input feature', fontsize=24)
        plt.ylabel('Relative importance', fontsize=24)
        plt.show()
        return indices
    
    
    def dependence(self, forest, train, feature_set):
        print "******************this is the output of dependences of features"
        fig, axs = plot_partial_dependence(forest, train, features=feature_set
                                           )
        plt.show()
        
        
    def dependence3d(self, forest, train, feature_set):
        print "******************this is the output of dependences of features"
        fig = plt.figure()
        pdp, (x_axis, y_axis) = partial_dependence(forest, feature_set,
                                           X=train)
        XX, YY = np.meshgrid(x_axis, y_axis)
        Z = pdp.T.reshape(XX.shape)
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
        #------------------------------- ax.set_xlabel(names[target_feature[0]])
        #------------------------------- ax.set_ylabel(names[target_feature[1]])
        ax.set_zlabel('Partial dependence')
        #  pretty init view
        ax.view_init(elev=22, azim=122)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of house value on me12dian age and '
                    'average occupancy')
        plt.subplots_adjust(top=0.9)
        
        plt.show()
        
        
        
    def str_float(self, x):
        tn = []
        fp = []
        fn = []
        tp = []
        
        for j in range(0, len(x)):
            raw_met = x[j]
            raw_met1 = np.array(raw_met)
    
            raw_met2 = np.str(raw_met1)
            raw_met3 = raw_met2.split()
            raw_met4 =np.str(raw_met3)
    
            raw_met5 =re.sub(r'[^0-9, .]', '', raw_met4)
            raw_met6 = re.sub(r'[,]', '', raw_met5)
    
            raw_met7= raw_met6.split()
            tn.append(np.float(raw_met7[0]))
            fp.append(np.float(raw_met7[1]))
            fn.append(np.float(raw_met7[2]))
            tp.append(np.float(raw_met7[3]))
            
        return pd.DataFrame({'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp})
    
    def precision(self, x):
        tn = x['tn']
        tp = x['tp']
        fn = x['fn']
        fp = x['fp']
        
        return tp/(tp + fp)
    
    def recall(self, x):  
        tn = x['tn']
        tp = x['tp']
        fn = x['fn']
        fp = x['fp']
        return tp / (tp + fn)    
    
    def accuracy(self, x):
        tn = x['tn']
        tp = x['tp']
        fn = x['fn']
        fp = x['fp']
        
        return (tp+tn)/(tp + fp + tn + fn)
    
    def f1score(self, x):
        
        precision = training.precision(self,x)
        recall = training.recall(self, x)  
        
        return 2*(precision * recall)/(precision + recall)

class test():    
    def __init__(self):
        print "*******************************************"
        
    def testforest_score(self, test, testlabel,forest):

          
        outputtest= forest.predict(test) 
        accuracytrain = accuracy_score(testlabel, outputtest)
        return confusion_matrix(testlabel, outputtest)
    
        #return accuracytrain
    
    def testforest_R(self, test, testlabel, forest):
        outputtest= forest.predict(test) 
        return r2_score(testlabel, outputtest),mean_squared_error(testlabel, outputtest)
                   
    
    
    def testforest_confu(self, test, testlabel,forest):
        outputtest= forest.predict(test) 
        accuracytrain = accuracy_score(testlabel, outputtest)
        #----------------------------------- print "The size of the test set is"
        #------------------------------------------------- print  np.shape(test)
#------------------------------------------------------------------------------ 
        # print "The accuracy for the test set is %r" %accuracytrain, "and the confusion matrix is"
        #-------------------------- print confusion_matrix(outputtest,testlabel)
        #------------------------------------- #output the classification report
        #-------------------- print classification_report(testlabel, outputtest)
        #generate probability
        output_proba=forest.predict_proba(test)
        out_perfor={'Classprob0':output_proba[:,0],'Classprob1':output_proba[:,1],
                    'Classprob2':output_proba[:,1],'output':outputtest,'target':testlabel}
        outframe=DataFrame(out_perfor)
 #       print accuracytrain
#        print outframe
        # save the outprobability
#        outframe.to_csv(r'D:\allprob.csv', header=0)

#        return outputtest
#        return (outframe)

#        print confusion_matrix(outputtest,testlabel)
        return  accuracytrain
    
    
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
                             verticalalignment='center', fontsize=40, color='#000000', fontweight='bold')
        
        cb = fig.colorbar(res)
        cb.ax.tick_params(labelsize=14)

    
        plt.xticks(np.arange(2), ['Unfailed','Failed'], fontsize=20)
        plt.yticks(np.arange(2), ['Unfailed','Failed'], fontsize=20, rotation=90)
        plt.xlabel('Predicted class', fontsize=24)
        plt.ylabel('True class', fontsize = 24)
        plt.show()
    
    def plot_gridsearch(self,x, aspect = 3):
        scores=np.array(x)
        scores=scores[:, 2:].T
        #    print scores
        #scores= scores[:,5:]
        print np.shape(scores)
        
        #    print np.arange(100,2010,20)
        
        figsize(16,5)
        fig, ax = plt.subplots(1,1)
        cax = ax.imshow(scores, interpolation='none', origin='highest',
                        cmap=plt.cm.coolwarm, aspect=aspect)
        
        plt.grid(b=True, which='x', color='white',linestyle='-')
        
        plt.xlim(0,96)
        
        plt.xticks((-0.5,20,45,70, 96.5), (100,500,1000,1500,2000), fontsize = 20)        
#        plt.xticks(np.linspace(0,194,10), int(np.linspace(100,4000,10)), fontsize = 20)
        plt.yticks(np.arange(0,11,1), np.arange(1,12,1), fontsize = 20)
        plt.xlabel('Number of trees',fontsize = 24)
        plt.ylabel('Number of features', fontsize = 24)
        ax.yaxis.grid(False,'major')
        ax.xaxis.grid(False, 'major')
        #cb = fig.colorbar(cax, orientation='horizontal', pad = 0.15, shrink=1, aspect=50)
        #cb = fig.colorbar(cax, orientation='vertical', shrink=2, aspect=2)
        cb = fig.colorbar(cax, orientation='vertical',shrink=1, aspect=20)
        cb.ax.tick_params(labelsize=14)
        #----------------------------------------------------------- scores=np.array(df)
        #------------------------------------------------------ scores=scores[:, 2:13].T
        #-------------------- plt.imshow(scores, interpolation='None', origin='highest',
                      #--------------------------------- cmap=plt.cm.coolwarm, aspect=3)
        #------------------------------------------------------------------------------ 
        #-------------- cb = plt.colorbar(orientation='horizontal', shrink=1, aspect=50)
        #---------------------------------------------- #sns.axes_style(axes.grid=False)
        #-------------------------------------------------------------------- plt.show()
        plt.tight_layout()
        plt.show()

