import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing, cross_validation, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from matplotlib.scale import LogScale
from pandas.core.frame import DataFrame
from sklearn import svm
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
from sklearn.svm import SVR
from sklearn import metrics
import re
from sklearn.cross_validation import cross_val_score
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import mean_absolute_error



#===============================================================================
# def metric_scores(estimator, testt, testlabelt):
#     
#       
#        y_pred = estimator.predict(testt)
#        #secret_cm.append(accuracy_score(testlabelt, y_pred))
#        secret_cm.append( metrics.confusion_matrix(testlabelt, y_pred).flatten())
#        
#        return accuracy_score(testlabelt, y_pred)
#===============================================================================
   

class training_manCV():
    secret_cm=[]
    
    def __init__(self):
        
        self.secret_cm=[]
        self.secret_score=[]
        
 
    def metric_scores(self, estimator, testt, testlabelt):
        
        
        y_pred = estimator.predict(testt)
        #secret_cm.append(accuracy_score(testlabelt, y_pred))
        training_manCV.secret_cm.append( metrics.confusion_matrix(testlabelt, y_pred).flatten())
        
        #print training_manCV.secret_cm
        training_manCV.secret_score.append( accuracy_score(testlabelt, y_pred))
        return accuracy_score(testlabelt, y_pred) 
   
    
    
    def str_float (self, x):
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
    
    
    def train_gene(self, train, trainlabel, seed, Cmin, Cmax, numC, rmin, rmax, numr, degree=3):
        
        C_range=np.logspace(Cmin, Cmax, num=numC, base=2,endpoint= True)
        gamma_range=np.logspace(rmin, rmax, num=numr, base=2,endpoint= True)
        
        
        paramgrid = {"kernel":[seed],
            "C":C_range,
            "gamma":gamma_range,
            "degree":[3]            
            }
        training_manCV.secret_score=[]
        ev = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring=training_manCV().metric_scores,
                                   cv=10,
                                   verbose=True,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   tournament_size=10,
                                   generations_number=100)
        ev.fit(train, trainlabel)
        
        print training_manCV.secret_cm
        print np.shape(training_manCV.secret_cm)
        print training_manCV.secret_score
        print np.shape(training_manCV.secret_score) 
        print ev.best_score_, ev.best_params_                
    
    
    
    def trainSVC (self, train, trainlabel, seed, Cmin, Cmax, numC, rmin, rmax, numr, degree=3):
        C_range=np.logspace(Cmin, Cmax, num=numC, base=2,endpoint= True)
        gamma_range=np.logspace(rmin, rmax, num=numr, base=2,endpoint= True)
        
        svc = SVC(kernel=seed)
#        mean_score=[]
        df_C_gamma= DataFrame({'gamma_range':gamma_range})
#        df_this = DataFrame({'gamma_range':gamma_range})
        count = 0 
        for C in C_range:    
            score_C=[]    
#            score_C_this = []
            count=count+1
            for gamma in gamma_range:
                
                training_manCV.secret_cm=[]     
                training_manCV.secret_score=[]      
                svc.C = C
                svc.gamma = gamma
                svc.degree = degree
                this_scores = cross_val_score(svc, train, trainlabel, scoring=training_manCV().metric_scores, cv=10, n_jobs=1)
                
       

                df_raw0 = DataFrame({'cm':training_manCV.secret_cm})
               
                
                score_C.append(np.mean(df_raw0['cm'].tail(10)))

               #score_C_this.append(np.mean(this_scores))
            print np.mean(this_scores) 
            print "%r cycle finished, %r left" %(count, numC-count)
            df_C_gamma[C]= score_C
            #df_this[C] = score_C_this 
        
        
        return df_C_gamma  


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



class training_classify(object):
    def __init__(self):
        print "This is for training set**************************************"
        
        
    def trainSVC(self, train, trainlabel, seed, Cmin, Cmax, num, plot=False):
        C_range=np.logspace(Cmin, Cmax, num=num, base=2 ,endpoint=True)
        gamma_range = C_range
        
        cv=10
        #cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(gamma=gamma_range,C=C_range) 
        grid = GridSearchCV(SVC(kernel=seed), param_grid=param_grid, cv=cv)
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
                    x, y = [self.vmin, self.midpoint, self.vmax], [0.51, 0.63, 0.76]
                    return np.ma.masked_array(np.interp(value, x, y))

            fig, ax = plt.subplots(figsize=(8, 6))
            #plt.figure(figsize=(8, 6))
            
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            
#            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.coolwarm)
            
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.coolwarm,
                       norm=MidpointNormalize(vmin=0.5, midpoint=0.6))            
            
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            
            
            #plt.xticks(np.arange(len(g_range)), g_range, rotation=45)#np.arrange sets the range of ticks
            #plt.yticks(np.arange(len(C_range)), C_range)
           # ax.xaxis.set_major_locator(MultipleLocator(1))
            #ax.xaxix.set_major_formatter(FormatStrFormatter('%d'))           
            plt.show()
            return (bestmodel, scores)
        
        else:
                    
            return (bestmodel, scores)        
            
    
        
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
        degree_range=range(3,4,1)
        cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(degree=degree_range, gamma=gamma_range, C=C_range) 
        grid = GridSearchCV(SVC('poly', coef0=1), param_grid=param_grid, cv=cv)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        #=======================================================================
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))        
        scores = DataFrame(scores)
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
    
    def svmsigmoid(self, train, trainlabel, Cmin, Cmax, num, base=2, plot=False): #default crossvalidation 10-fold
        C_range=np.logspace(Cmin, Cmax, num=num, base=base)
        gamma_range=np.logspace(Cmin, Cmax, num=num, base=base)
        
        cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(gamma=gamma_range, C=C_range) 
        grid = GridSearchCV(SVC('sigmoid', coef0=0), param_grid=param_grid, cv=cv)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        #=======================================================================
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))        
        scores = DataFrame(scores)
        #=======================================================================
        bestmodel= svm.SVC(kernel='sigmoid', gamma=1/12, coef0= 0,  C= grid.best_params_["C"]).fit(train,trainlabel)
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
            
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.oolwarm,
                       norm=MidpointNormalize(vmin=0.4, midpoint=0.55))
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            
            plt.xticks(np.arange(len(gamma_range)), gamma_range)#np.arrange sets the range of ticks
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
            
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.coolwarm)
            
#            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.coolwarm,
#                       norm=MidpointNormalize(vmin=0.6, midpoint=0.75))            
            
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            
           

            
            #plt.xticks(np.arange(len(g_range)), g_range, rotation=45)#np.arrange sets the range of ticks
            #plt.yticks(np.arange(len(C_range)), C_range)
           # ax.xaxis.set_major_locator(MultipleLocator(1))
            #ax.xaxix.set_major_formatter(FormatStrFormatter('%d'))           
            plt.show()
            return (bestmodel, scores)
        
        else:
                    
            return (bestmodel, scores)

class training_regress(object):
    def __init__(self):
        print "This is for training set**************************************"
        
    def svmlinear(self, train, trainlabel, Cmin, Cmax, num, base=2): #default crossvalidation 10-fold
        C_range=np.logspace(Cmin, Cmax, num=num, base=base)
        #gamma_range=np.logspace(-15,15,num=11,base=2.0)
        #cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
        param_grid = dict(C=C_range) 
        grid = GridSearchCV(SVR(kernel='linear'), param_grid=param_grid, cv=10)
        grid.fit(train, trainlabel) 
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))  
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_range))
        scores = DataFrame(scores)
        bestmodel= SVR(kernel='linear',C= grid.best_params_["C"]).fit(train,trainlabel)
        return (bestmodel, scores)
    
    def svmrbf(self, train, trainlabel, Cmin, Cmax, gmin, gmax, num, base=2, plot=False): #default crossvalidation 10-fold
            C_range=np.logspace(Cmin, Cmax, num=num, base=base)
            gamma_range=np.logspace(gmin,gmax,num=num,base=base)
            ep_range =np.logspace(Cmin, Cmax, num=num, base=base)
            #cv = StratifiedShuffleSplit(trainlabel, n_iter=10, test_size=0.1, random_state=0)
            param_grid = dict(gamma=gamma_range,C=C_range, epsilon=ep_range) 
            grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
            grid.fit(train, trainlabel) 
            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))  
           # scores = [x[1] for x in grid.grid_scores_]
           # scores = np.array(scores).reshape(len(C_range), len(gamma_range))        
           # scores = DataFrame(scores)
            bestmodel= svm.SVR(kernel='rbf', gamma =grid.best_params_["gamma"], C= grid.best_params_["C"], epsilon= grid.best_params_["epsilon"]).fit(train,trainlabel)
            
            #--------------------------------------------------- if plot== True:
                #--------------------------- class MidpointNormalize(Normalize):
                    # def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                        #------------------------------ self.midpoint = midpoint
                        #------------ Normalize.__init__(self, vmin, vmax, clip)
#------------------------------------------------------------------------------ 
                    #--------------------- def __call__(self, value, clip=None):
                        # x, y = [self.vmin, self.midpoint, self.vmax], [0.4, 0.6, 0.82]
                        #----- return np.ma.masked_array(np.interp(value, x, y))
#------------------------------------------------------------------------------ 
                #------------------------ fig, ax = plt.subplots(figsize=(8, 6))
                #----------------------------------- #plt.figure(figsize=(8, 6))
#------------------------------------------------------------------------------ 
                # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
#------------------------------------------------------------------------------ 
                #-- plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                           #-------------------------- norm=MidpointNormalize())
                #------------------------------------------- plt.xlabel('gamma')
                #----------------------------------------------- plt.ylabel('C')
                #------------------------------------------------ plt.colorbar()
                
               
                #------------------------------ g_range=np.linspace(-10, 10, 11)
                #------------------------------------------------- print g_range
#------------------------------------------------------------------------------ 
                # plt.xticks(np.arange(len(g_range)), g_range, rotation=45)#np.arrange sets the range of ticks
                #------------------ plt.yticks(np.arange(len(C_range)), C_range)
                #---------------- ax.xaxis.set_major_locator(MultipleLocator(1))
                #------- #ax.xaxix.set_major_formatter(FormatStrFormatter('%d'))
                #---------------------------------------------------- plt.show()

                        
            return (bestmodel)

class test(object):   
    def __init__(self):
        print "This is for test set**************************************"
               
      
    def test_classification(self, test, testlabel,bestmodel):
#        bestmodel=bestmodel
        outputtest = bestmodel.predict(test)
        accuracytest = accuracy_score(testlabel, outputtest)
        print "The accuracy for the test set is %r" %accuracytest, "and the confusion matrix is"
        print confusion_matrix(outputtest,testlabel)
        print classification_report(testlabel, outputtest)
#        probaout=bestmodel.predict_prob(test)
#       probaout= DataFrame(probaout)
#        print probaout
        return outputtest
        
    def test_regression(self, test, testlabel,bestmodel):
        
        outputtest = bestmodel.predict(test)
 #       print outputtest
        MAE = mean_absolute_error(testlabel, outputtest)

        print "The MAE for the test set is %r" %MAE       
        return outputtest
        