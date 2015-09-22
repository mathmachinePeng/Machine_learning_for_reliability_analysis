from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import RFclass
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
import TAlogistic as tl
import cPickle, theano
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import TAmlp as mlp
import TAdbn as dbn
import TAsda as sda
# process data into scaled training and testing

df =pd.read_csv('/home/peng/new160half.csv', header=0)
#data=standardprocess()
p= Preprocessdata.standardprocess()

train, trainlabel, test, testlabel = p.scaledivd(df, 0.7)
len = len(train)
valid = train[80:len, :]
validlabel = trainlabel[80:len]
train=train[0:80,:]
trainlabel = trainlabel[0:80]
#------------------------------------------- trainlabel=[trainlabel, trainlabel]
#----------------------------------------- trainlabel = np.transpose(trainlabel)
dataset=[(train, trainlabel), (valid, validlabel), (test, testlabel)]

dbn.test_DBN(pretraining_epochs= 100,training_epochs=1000, pretrain_lr=0.0001, finetune_lr= 0.15,dataset=dataset, batch_size=3)


#sda.test_SdA(dataset= dataset, training_epochs=1000, batch_size=10)


# patience matters change from 10 to 10*batch size, and test score from 0.5 to 0.75, the 
#best patience is from 129, hidden layer size = [10, 10], corrupt level = [.1,.2 ], 0.81

 






"""This mlp"""
#===============================================================================
# outputtest = mlp.test_mlp(learning_rate=0.0000001,L2_reg=0.001, n_hidden=50, batch_size=3,dataset=dataset)
# 
# print "this is the final"
# print outputtest
# 
# print accuracy_score(testlabel, outputtest)
#===============================================================================
#----------------------------------- print accuracy_score(testlabel, outputtest)
#mlp.predict(test)

# best result  Best validation score of 23.333333 % obtained at iteration 20, with test performance 18.750000 %
# mlp(learning_rate=0.0000001,L2_reg=0.1, n_hidden=10, batch_size=3,dataset=dataset)

"""This is SGD Logistic regression"""
#----------- tl.free_sgd_optimization_mnist(learning_rate=0.001, n_epochs =1001,
                               #----------------- dataset=dataset, batch_size=3)
                               




##############Test the theano model
#-------------------------------------------------- outputtest=mlp.predict(test)
#---------------------------- print classification_report(testlabel, outputtest)
#--------------------------------- print confusion_matrix(testlabel, outputtest)
#----------------------------------- print accuracy_score(testlabel, outputtest)

""" THis is RF"""
# Train the model by RF

#--------------------------------------------------------- ff=RFclass.training()
#------------------------ forest= ff.trainforest('ext', train, trainlabel, 1000)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------- tt=RFclass.test()
#------------------------------------- LL=tt.testforest(test, testlabel, forest)

#ff.sensitivity(forest, 12)


""" This is SVM"""
# # Train with SVM
#===============================================================================
# svcc=mysvc.training()
# best, scores=svcc.svmlinear(train, trainlabel, -10, 10,41)
# # 
# # # Test with SVM
# svtt=mysvc.test()
# svtt.testsvm(test, testlabel, best)
#===============================================================================
 
"""This is SVM"""
#===============================================================================
# # # Train with SVM
# svcc=mysvc.training()
# best, scores=svcc.svmlinear(train, trainlabel, -10, 10,20)
# # 
# # # Test with SVM
# svtt=mysvc.test()
# svtt.testsvm(test, testlabel, best)
#===============================================================================
 
 
 
# Train with SVM
#===============================================================================
# svcc=mysvc.training()
# best, scores=svcc.svmrbf(train, trainlabel, -10, 10,-10, 10, 41, )
#  
# # Test with SVM
# svtt=mysvc.test()
# svtt.testsvm(test, testlabel, best)
#===============================================================================

#===============================================================================
# n_group = 5
# linearvalue=(76, 81.25, 86, 76, 81)
# polyvalue= (76, 73, 75, 72, 73)
# rbfvalue=(79, 83, 79, 92, 85)
# 
# fig, ax= plt.subplots()
# 
# index=np.arange(n_group)
# bar_width= 0.35
# 
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
# 
# rects1 = plt.bar(index, linearvalue, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  
#                  error_kw=error_config,
#                  label='linear')
# 
# rects2 = plt.bar(index + bar_width, rbfvalue, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  
#                  error_kw=error_config,
#                  label='polynomial')
# 
# rects2 = plt.bar(index + 2*bar_width, rbfvalue, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  
#                  error_kw=error_config,
#                  label='rbf')
# 
# plt.xlabel('Classification metrics')
# plt.ylabel('Percentage value (%)')
# #plt.title('Scores by group and gender')
# 
# ax.set_xticks(index+5*bar_width)
# plt.xticks(index + 1.5*bar_width, ('CV accuracy', 'Test accuracy', 'Precision', 'Recall', 'F1 score'))
# plt.legend()
# 
# plt.tight_layout()
# plt.show()
#===============================================================================