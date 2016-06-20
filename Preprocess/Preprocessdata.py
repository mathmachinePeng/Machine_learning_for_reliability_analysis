from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score, confusion_matrix
import csv
from sklearn.ensemble import RandomForestClassifier


class standardprocess():
    def __init__(self):
        print ("*************************************change*")
    
    def init_process(self,raw_data, ratio):
        X=np.array(raw_data)
        rows,dimens=np.shape(X)
        new_rows= int(rows*ratio)
        new_dimens= dimens-1       
        return(rows, dimens, new_rows, new_dimens)

    def sep_scale_divd(self, raw_data, ratio):
        a, b, aa, bb = self.init_process(raw_data, ratio)
        X_origin=np.array(raw_data)
        


        Train=X_origin[0:aa,0:bb]
        Trainlabel=X_origin[0:aa,bb]
        Test=X_origin[aa:a,0:bb]
        Testlabel=X_origin[aa:a,bb]   
        
        min_max_scaler=preprocessing.MinMaxScaler()
        
        Train= min_max_scaler.fit_transform(Train)        
        Test = min_max_scaler.transform(Test)  
        return(Train, Trainlabel, Test, Testlabel)


        
    def scaledivd(self, raw_data, ratio):
        a, b, aa, bb = self.init_process(raw_data, ratio)
        X=np.array(raw_data)
        min_max_scaler=preprocessing.MinMaxScaler()
        Xscale= min_max_scaler.fit_transform(X)

        Train=Xscale[0:aa,0:bb]
        Trainlabel=Xscale[0:aa,bb]
        Test=Xscale[aa:a,0:bb]
        Testlabel=Xscale[aa:a,bb]        
        return(Train, Trainlabel, Test, Testlabel)
    
    
    
    
    def noscale(self, raw_data, ratio):
        a, b, aa, bb = self.init_process(raw_data, ratio)
        X=np.array(raw_data)
        Train=X[0:aa,0:bb]
        Trainlabel=X[0:aa,bb]
        Test=X[aa:a,0:bb]
        Testlabel=X[aa:a,bb]        
        return(Train, Trainlabel, Test, Testlabel)
    
    def normalscale(self, raw_data, ratio):
        X=np.array(raw_data)
        rows, dimens, new_rows, new_dimens = self.init_process(raw_data, ratio)
        X=preprocessing.scale(X)
        Train=X[0:new_rows,0:new_dimens]
        Trainlabel=X[0:new_rows,new_dimens]
        Test=X[new_rows:rows,0:new_dimens]
        Testlabel=X[new_rows:rows,new_dimens]        
        return(Train, Trainlabel, Test, Testlabel)
    
    def noaction(self, raw_data): 
        X=np.array(raw_data)
        rows,dimens=np.shape(X)
        Train=X[:,0:dimens-1]
        Trainlabel=X[:,dimens-1]
        return Train, Trainlabel
    