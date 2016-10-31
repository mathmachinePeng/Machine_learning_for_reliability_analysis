'''
Created on 30 May 2016

@author: peng
'''
import numpy as np
import pandas as pd

data_path = '/home/peng/Documents/Project_C/QSAR_nlp/Codes_git/bob/tutorial/'
data_name = 'dsgdb7ae2.xyz'
data_one_name = 'single_try.xyz'

xyz = open (data_path + data_name, 'r')
line = xyz.readline()
print line

a = 0
if str.isalpha(line):
    a = a+1
    print a
