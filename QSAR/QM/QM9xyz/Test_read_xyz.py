'''
Created on 2 Jun 2016

@author: peng
'''
import numpy as np
import pandas as pd
import MDAnalysis as md 

path_133k = '/home/peng/Documents/Project_C/QSAR_nlp/Dataset_qm9/133k/dsgdb9nsd_'

#--------------------- db = md.coordinates.XYZ.XYZReader(path_133k+'000001.xyz')
#------------------------------------------------------------------------------ 
#----------------------------------------------------- all = list(m for m in db)
#------------------------------------------------------------------------------ 
#---------------------------------------------------------------------- print db

file = open(path_133k + '000001.xyz', 'r')
print file.read()
file.close()