'''
Created on 2 Jun 2016

@author: peng
'''
import pandas as pd
import numpy as np

path_paper = '/home/peng/Documents/Project_C/QSAR_nlp/Dataset_kaggle/2269153/ci500747n_si_002/'

path_web = '/home/peng/Documents/Project_C/QSAR_nlp/Dataset_kaggle/web/'


datasets_names = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 
                  'OX1', 'OX2', 'PGP', "PPB", 'RAT_F', 'TDI', 'THROMBIN']

for i in datasets_names:

    df = pd.read_csv(path_paper + '%s'%i+'_test_disguised.csv', header= 0)
    
    print df.shape
cx js

### Print the shape of web sets
#---------------------------------------------------- for i in np.arange(1, 16):
    # df = pd.read_csv(path_web + 'ACT%d'%i+'_competition_training.csv', header=0)
    #------------------------------------------------------------ print df.shape
 ####   
    



######compare the intersection of two sets : features 
#-------------------------------------- list0 = list(df.columns.values.tolist())
#------------------------------------------------------------------------------ 
#--------------------------------------------------------- print np.shape(list0)
#------------------------------------------------------------------------------ 
#------------------------------------- list1 = list(df1.columns.values.tolist())
#------------------------------------------------------------------------------ 
#--------------------------------------------------------- print np.shape(list1)
#------------------------------------------------------------------------------ 
#-------------------------- print np.shape(list(set(list0).intersection(list1)))


