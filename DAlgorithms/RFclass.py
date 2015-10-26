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
        
    def trainforest(self, seed, train, trainlabel, number_trees, number_features):
        seed_of_tree = {'rf': RandomForestClassifier(n_estimators= number_trees, max_features=number_features), 
                      'adb': AdaBoostClassifier(n_estimators= number_trees),
                      'bag': BaggingClassifier(n_estimators= number_trees),
                      'ext': ExtraTreesClassifier(n_estimators= number_trees, max_features=number_features),
                      'gbt': GradientBoostingClassifier(n_estimators= number_trees, max_features=number_features),
                      'bagging': RandomForestClassifier(n_estimators= number_trees, max_features=12)}
        rawforest=seed_of_tree[seed]
        forest=rawforest.fit(train,trainlabel)
        outputtrain= forest.predict(train)
        accuracytrain = accuracy_score(trainlabel, outputtrain)        
#        print "The size of the training set is %r , %r" %(np.shape(train)[0],np.shape(train)[1])
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
        plt.figure(figsize=(8,6.5))
        plt.bar(range(12), importances[indices],
            color="c", align="center")
         #------------------------------ plt.bar(range(12), importances[indices],
             #--------------------- color="c", yerr=std[indices], align="center")
        plt.xticks(range(n), indices+1, fontsize=14)
        plt.yticks(fontsize = 14)
        plt.xlim([-1, n])
#        plt.ylim([0.00,0.30])
        plt.xlabel('The input feature', fontsize=24)
        plt.ylabel('Relative importance', fontsize=24)
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
                    

                    ax.annotate(str(cm[x][y]), xy=(x, y),
                            horizontalalignment='center',
                             verticalalignment='center', fontsize=40, color='#00aaff', fontweight='bold')
        
        cb = fig.colorbar(res)
        cb.ax.tick_params(labelsize=14)

    
        plt.xticks(np.arange(2), ['Unfailed','Failed'], fontsize=20)
        plt.yticks(np.arange(2), ['Unfailed','Failed'], fontsize=20, rotation=90)
        plt.xlabel('True class', fontsize=24)
        plt.ylabel('Predicted class', fontsize = 24)
        plt.show()


