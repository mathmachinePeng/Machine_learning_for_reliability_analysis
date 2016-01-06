'''
Created on 23 Nov 2015

@author: peng
'''
import numpy as np
A = np.arange(1,10,1)
x= np.log(A)
v=x.diff()
print x
print v