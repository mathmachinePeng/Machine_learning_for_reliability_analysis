'''
Created on 2 Jun 2016

@author: peng
'''
import os,pickle,sys,numpy,nn,copy,scipy,scipy.io
import numpy as np
import pandas as pd

split = 5

dataset = scipy.io.loadmat('/home/peng/Documents/Project_C/QSAR_nlp/qm7.mat')
P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
X = dataset['X'][P]
T = dataset['T'][0,P]

print np.shape(P)
print np.shape(X)
print np.shape(T)

I,O = nn.Input(X),nn.Output(T)

print I