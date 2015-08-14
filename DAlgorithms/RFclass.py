from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier


class training(object):
    def __init__(self):
        print "This is for training set**************************************"
        
    def trainforest(self, seed, train, trainlabel, number_trees):
        seedoftree = {'rf': RandomForestClassifier(n_estimators= number_trees), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees)}
        rawforest=seedoftree[seed]
        forest=rawforest.fit(train,trainlabel)
        outputtrain= forest.predict(train)
        accuracytrain = accuracy_score(trainlabel, outputtrain)        
        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1]) 
        print "The method is %r" %seed
        print "The accuracy for the training set is %r" %accuracytrain, "and the confusion matrix is"
        print confusion_matrix(outputtrain,trainlabel)
        return (forest)
    
    def sensitivity(self, forest, n):
        print "**************************************"
        print(forest.feature_importances_)
        importances=forest.feature_importances_
        std=np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices=np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(12):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.figure()
        plt.bar(range(12), importances[indices],
               color="c", yerr=std[indices], align="center")
        plt.xticks(range(n), indices)
        plt.xlim([-1, n])
        plt.xlabel('The input feature')
        plt.ylabel('Relative importance')
        plt.show()
    
    

        

class test():    
    def __init__(self):
        print "*******************************************"
        
    def testforest(self, test, testlabel,forest):
        outputtest= forest.predict(test) 
        accuracytrain = accuracy_score(testlabel, outputtest)
        print "The size of the test set is"     
        print  np.shape(test)
        print "The accuracy for the test set is %r" %accuracytrain, "and the confusion matrix is"
        print confusion_matrix(outputtest,testlabel)
        print classification_report(testlabel, outputtest)
        # generate probability
        outputproba=forest.predict_proba(test)       
        outperfor={'prob0':outputproba[:,0],'prob1':outputproba[:,1],'output':outputtest,'target':testlabel}
        outframe=DataFrame(outperfor)
        #print outframe
        #outframe.to_csv(r'D:\allprob.csv', header=0)
        return (outframe)
        
         


