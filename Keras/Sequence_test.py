'''
Created on 23 Mar 2016

@author: peng
'''

#------------------------------------------- from keras.models import Sequential
#------------------------------- from keras.layers.core import Dense, Activation
#--------------------------------------- from keras.layers.recurrent import LSTM
#------------------------------------------------------------------------------ 
#------------------------------------------------------------ in_out_neurons = 2
#---------------------------------------------------------- hidden_neurons = 300
#------------------------------------------------------------------------------ 
#----------------------------------------------------------- import pandas as pd
#----------------------------------------------------- from random import random
#------------------------------------------------------------------------------ 
#---------------------- flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
#------------------------------------ pdata = pd.DataFrame({"a":flow, "b":flow})
#---------------------------------------------------- pdata.b = pdata.b.shift(9)
#------------------------------- data = pdata.iloc[10:] * random()  # some noise
#------------------------------------------------------------------------------ 
#------------------------------------------------------------ import numpy as np
#------------------------------------------------------------------------------ 
#------------------------------------------- def _load_data(data, n_prev = 100):
    #----------------------------------------------------------------------- """
    #--------------------------------------------- data should be pd.DataFrame()
    #----------------------------------------------------------------------- """
#------------------------------------------------------------------------------ 
    #------------------------------------------------------- docX, docY = [], []
    #----------------------------------------- for i in range(len(data)-n_prev):
        #------------------------ docX.append(data.iloc[i:i+n_prev].as_matrix())
        #-------------------------- docY.append(data.iloc[i+n_prev].as_matrix())
    #----------------------------------------------------- alsX = np.array(docX)
    #----------------------------------------------------- alsY = np.array(docY)
#------------------------------------------------------------------------------ 
    #--------------------------------------------------------- return alsX, alsY
#------------------------------------------------------------------------------ 
#-------------------------------------- def train_test_split(df, test_size=0.1):
    #----------------------------------------------------------------------- """
    #----------------------- This just splits data to training and testing parts
    #----------------------------------------------------------------------- """
    #----------------------------------- ntrn = round(len(df) * (1 - test_size))
#------------------------------------------------------------------------------ 
    #---------------------------- X_train, y_train = _load_data(df.iloc[0:ntrn])
    #------------------------------- X_test, y_test = _load_data(df.iloc[ntrn:])
#------------------------------------------------------------------------------ 
    #------------------------------- return (X_train, y_train), (X_test, y_test)
#------------------------------------------------------------------------------ 
# (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
#------------------------------------------------------------------------------ 
#---------------------------------------------------------- model = Sequential()
#-------------------------------- model.add(LSTM(5, 300, return_sequences=True))
#------------------------------ model.add(LSTM(300, 500, return_sequences=True))
#------------------------------------------------------- model.add(Dropout(0.2))
#----------------------------- model.add(LSTM(500, 200, return_sequences=False))
#------------------------------------------------------- model.add(Dropout(0.2))
#------------------------------------------------------ model.add(Dense(200, 3))
#----------------------------------------------- model.add(Activation("linear"))
#----------------- model.compile(loss="mean_squared_error", optimizer="rmsprop")
#------------------------------------------------------------------------------ 
# model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)




############################################ Example2#################
#------------------------------------------------------------------------------ 
#------------------------------------------- from keras.models import Sequential
#------- from keras.layers.core import TimeDistributedDense, Activation, Dropout
#---------------------------------------- from keras.layers.recurrent import GRU
#------------------------------------------------------------ import numpy as np
#------------------------------------------------------------------------------ 
#--------------------------------------------- def _load_data(data, steps = 40):
    #------------------------------------------------------- docX, docY = [], []
    #--------------------------------- for i in range(0, data.shape[0]/steps-1):
        #------------------------------ docX.append(data[i*steps:(i+1)*steps,:])
        #---------------------- docY.append(data[(i*steps+1):((i+1)*steps+1),:])
    #----------------------------------------------------- alsX = np.array(docX)
    #----------------------------------------------------- alsY = np.array(docY)
    #--------------------------------------------------------- return alsX, alsY
#------------------------------------------------------------------------------ 
#----------------------------------- def train_test_split(data, test_size=0.15):
    #------------------ #    This just splits data to training and testing parts
    #---------------------------------------------------- X,Y = _load_data(data)
    #-------------------------------- ntrn = round(X.shape[0] * (1 - test_size))
    #--------------------------------- perms = np.random.permutation(X.shape[0])
    # X_train, Y_train = X.take(perms[0:ntrn],axis=0), Y.take(perms[0:ntrn],axis=0)
    #-- X_test, Y_test = X.take(perms[ntrn:],axis=0),Y.take(perms[ntrn:],axis=0)
    #------------------------------- return (X_train, Y_train), (X_test, Y_test)
#------------------------------------------------------------------------------ 
#-------------------------------------- np.random.seed(0)  # For reproducability
#---------------------- data = np.genfromtxt('closingAdjLog.csv', delimiter=',')
# (X_train, y_train), (X_test, y_test) = train_test_split(np.flipud(data))  # retrieve data
#---------------------------------------------------------- print "Data loaded."
#------------------------------------------------------------------------------ 
#----------------------------------------------------------- in_out_neurons = 20
#---------------------------------------------------------- hidden_neurons = 200
#------------------------------------------------------------------------------ 
#---------------------------------------------------------- model = Sequential()
# model.add(GRU(hidden_neurons, input_dim=in_out_neurons, return_sequences=True))
#------------------------------------------------------- model.add(Dropout(0.2))
#------------------------------- model.add(TimeDistributedDense(in_out_neurons))
#----------------------------------------------- model.add(Activation("linear"))
#----------------- model.compile(loss="mean_squared_error", optimizer="rmsprop")
#------------------------------------------------------- print "Model compiled."
#------------------------------------------------------------------------------ 
#---------------------------------------------------- # and now train the model.
# model.fit(X_train, y_train, batch_size=30, nb_epoch=200, validation_split=0.1)
#--------------------------------------------- predicted = model.predict(X_test)
# print np.sqrt(((predicted - y_test) ** 2).mean(axis=0)).mean()  # Printing RMSE


###################################################Example3#############
import pandas as pd  
from random import random

###generate two sequences####### 

#---------------------- flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
#------------------------------------ pdata = pd.DataFrame({"a":flow, "b":flow})
#---------------------------------------------------- pdata.b = pdata.b.shift(9)
#------------------------------- data = pdata.iloc[10:] * random()  # some noise

#data.to_csv('raw_data_for_test.csv', header=True)

########check the raw_data and plot it ####

#---------------------------------- data = pd.read_csv('raw_data.csv', header=0)
#-------------------------------------------------------------------- print data
#------------------------------------------------------ import matplotlib as mpl
#----------------------------------------------- import matplotlib.pyplot as plt
#------------------------------------------------------------------------------ 
#------------------------------------------------- plt.plot(data['a'],data['b'])
#-------------------------------------------------------------------- plt.show()

##############################################
###########read the input data and turn it to proper format##############
data = pd.read_csv('raw_data.csv', header=0)


import numpy as np

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))


    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = train_test_split(data.iloc[0:20000])  # retrieve data

#print np.shape(X_train)[0]
#print np.shape(y_test)[0]

##################################################################################################


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_neurons = 2
out_neurons = 2
hidden_neurons = 20

model = Sequential()
model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False))

model.add(Dense(output_dim=out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


model.fit(X_train, y_train, batch_size=np.shape(X_train)[0], nb_epoch=2, validation_split=0.05)

predicted = model.predict(X_test,batch_size=np.shape(y_test)[0])
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
#print rmse
##################################
# and maybe plot it

#------------------------- pd.DataFrame(predicted[:100]).to_csv("predicted.csv")
#------------------------------ pd.DataFrame(y_test[:100]).plot("test_data.csv")


