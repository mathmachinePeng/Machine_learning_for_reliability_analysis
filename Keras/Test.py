'''
Created on 24 Mar 2016

@author: peng
'''
import numpy as np
import pandas as pd

df=pd.DataFrame({'a':np.arange(0,10,1), 'b':np.arange(10,20,1)})

print df.iloc[0:2], df.iloc[2]