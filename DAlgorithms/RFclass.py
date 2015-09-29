"""This is the class containing ensemble methods like random forestg, bagging and boosting 
which contains a lot single learners and output the final result based on the voting of every
single learner

Peng Jiang 
27/09/2015 Happy middle moon festival"""


from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D


class training(object):
    def __init__(self):
        print "This is for training set**************************************"
        
    def trainforest(self, seed, train, trainlabel, number_trees):
        seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees),
                      'gbt': GradientBoostingClassifier(n_estimators= number_trees),
                      'bagging': RandomForestClassifier(n_estimators= number_trees, max_features=12)}
        rawforest=seed_of_tree[seed]
        forest=rawforest.fit(train,trainlabel)
        outputtrain= forest.predict(train)
        accuracytrain = accuracy_score(trainlabel, outputtrain)        
        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1])
        #---------------------------------------- print "The method is %r" %seed
        # print "The accuracy for the training set is %r" %accuracytrain, "and the confusion matrix is"
        #------------------------ print confusion_matrix(outputtrain,trainlabel)
        return (forest)
    
    def importance(self, forest, n):
        print "************************this is the output of relative importance**************"
        #print(forest.feature_importances_)
        importances=forest.feature_importances_
        #return importances
        #std=np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices=np.argsort(importances)[::-1]
        print indices
        print("Feature ranking:")
        for f in range(12):
              print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.figure()
        plt.bar(range(12), importances[indices],
            color="c", align="center")
         #------------------------------ plt.bar(range(12), importances[indices],
             #--------------------- color="c", yerr=std[indices], align="center")
        plt.xticks(range(n), indices+1)
        plt.xlim([-1, n])
        plt.xlabel('The input feature', fontsize=16)
        plt.ylabel('Relative importance', fontsize=16)
        plt.show()
    
    
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
        plt.suptitle('Partial dependence of house value on median age and '
                    'average occupancy')
        plt.subplots_adjust(top=0.9)
        
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
        #output the classification report
        print classification_report(testlabel, outputtest)
        #generate probability
        output_proba=forest.predict_proba(test)
        out_perfor={'Classprob0':output_proba[:,0],'Classprob1':output_proba[:,1],
                    'Classprob2':output_proba[:,1],'output':outputtest,'target':testlabel}
        outframe=DataFrame(out_perfor)
#        print outframe
        # save the outprobability
#        outframe.to_csv(r'D:\allprob.csv', header=0)
#        return outputtest
#        return (outframe)
        return  confusion_matrix(outputtest,testlabel)
        
    def plot_confusion_matrix(self, cm):
        norm_conf = []
        for i in cm:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)
        
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.summer, 
                        interpolation='nearest')
        
        width = len(cm)
        height = len(cm[0])
        
        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(cm[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=40, color='#00aaff', fontweight='bold')
        
        cb = fig.colorbar(res)
    
        plt.xticks(np.arange(2), ['Unfailed','Failed'], fontsize=14)
        plt.yticks(np.arange(2), ['Unfailed','Failed'], fontsize=14, rotation=90)
        plt.xlabel('Predicted class', fontsize=16)
        plt.ylabel('True class', fontsize = 16)
        plt.show()

