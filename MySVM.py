import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import Preprocessdata1 as p
from matplotlib.scale import LogScale
from pandas.core.frame import DataFrame
from sklearn import svm
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 

class training(object):
    def __init__(self):
        print "This is for training set**************************************"
        
    def svmlinear(self, train, trainlabel, Cmin, Cmax, num, base=2): #default crossvalidation 10-fold
        C_range=np.logspace(Cmin, Cmax, num=num, base=base)
        #gamma_range=np.logspace(-15,15,num=11,base=2.0)
        cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(C=C_range) 
        grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range))
        scores = DataFrame(scores)
        bestmodel= svm.SVC(kernel='linear',C= grid.best_params_["C"]).fit(train,trainlabel)
        return (bestmodel, scores)
    
    
    
    
    def svmpoly(self, train, trainlabel, Cmin, Cmax, num, base=2, plot=False): #default crossvalidation 10-fold
        C_range=np.logspace(Cmin, Cmax, num=num, base=base)
        gamma_range=np.logspace(Cmin, Cmax, num=num, base=base)
        degree_range=range(1,6,1)
        cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(degree=degree_range, gamma=gamma_range, C=C_range) 
        grid = GridSearchCV(SVC('poly', coef0=1), param_grid=param_grid, cv=cv)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        #=======================================================================
        # scores = [x[1] for x in grid.grid_scores_]
        # scores = np.array(scores).reshape(len(C_range), len(gamma_range), len(degree_range))        
        # scores = DataFrame(scores)
        #=======================================================================
        bestmodel= svm.SVC(kernel='poly', gamma=grid.best_params_["gamma"], coef0= 1, degree =grid.best_params_["degree"], C= grid.best_params_["C"]).fit(train,trainlabel)
#        print grid.best_params_
        if plot== True:
            class MidpointNormalize(Normalize):
                def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                    self.midpoint = midpoint
                    Normalize.__init__(self, vmin, vmax, clip)
            
                def __call__(self, value, clip=None):
                    x, y = [self.vmin, self.midpoint, self.vmax], [0.4, 0.6, 0.82]
                    return np.ma.masked_array(np.interp(value, x, y))

            plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                       norm=MidpointNormalize(vmin=0.4, midpoint=0.55))
            plt.xlabel('degree')
            plt.ylabel('C')
            plt.colorbar()
            
            plt.xticks(np.arange(len(degree_range)), degree_range)#np.arrange sets the range of ticks
            plt.yticks(np.arange(len(C_range)), C_range)           
            plt.show()
            return (bestmodel, grid.best_score_)
        
        else:
                    
            return (bestmodel, grid.best_score_)
    
    
    
    def svmrbf(self, train, trainlabel, Cmin, Cmax, gmin, gmax, num, base=2, plot=False): #default crossvalidation 10-fold
        C_range=np.logspace(Cmin, Cmax, num=num, base=base)
        gamma_range=np.logspace(gmin,gmax,num=num,base=base)
        cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(gamma=gamma_range,C=C_range) 
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))        
        scores = DataFrame(scores)
        bestmodel= svm.SVC(kernel='rbf', gamma =grid.best_params_["gamma"], C= grid.best_params_["C"]).fit(train,trainlabel)
        
        if plot== True:
            class MidpointNormalize(Normalize):
                def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                    self.midpoint = midpoint
                    Normalize.__init__(self, vmin, vmax, clip)
            
                def __call__(self, value, clip=None):
                    x, y = [self.vmin, self.midpoint, self.vmax], [0.4, 0.6, 0.82]
                    return np.ma.masked_array(np.interp(value, x, y))

            fig, ax = plt.subplots(figsize=(8, 6))
            #plt.figure(figsize=(8, 6))
            
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                       norm=MidpointNormalize(vmin=0.4, midpoint=0.55))
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            
           
            g_range=np.linspace(-10, 10, 11)
            print g_range
            
            plt.xticks(np.arange(len(g_range)), g_range, rotation=45)#np.arrange sets the range of ticks
            plt.yticks(np.arange(len(C_range)), C_range)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            #ax.xaxix.set_major_formatter(FormatStrFormatter('%d'))           
            plt.show()
            return (bestmodel, scores)
        
        else:
                    
            return (bestmodel, scores)


class test(object):   
    def __init__(self):
        print "This is for test set**************************************"
               
      
    def testsvm(self, test, testlabel,bestmodel):
        bestmodel=bestmodel
        outputtest = bestmodel.predict(test)
        accuracytest = accuracy_score(testlabel, outputtest)
        print "The accuracy for the test set is %r" %accuracytest, "and the confusion matrix is"
        print confusion_matrix(outputtest,testlabel)
        print classification_report(testlabel, outputtest)
#        probaout=bestmodel.predict_prob(test)
#       probaout= DataFrame(probaout)
#        print probaout
        return outputtest
        

        