import os,pickle,sys,numpy,nn,copy,scipy,scipy.io
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import sgd
from keras.layers.recurrent import LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
import json
import timeit
from sklearn import preprocessing
import pandas as pd

start = timeit.default_timer()
# --------------------------------------------
# Parameters
# --------------------------------------------



## import data
root_folder_path = '/home/peng/Desktop/ Pynotebooks/Link to Machine_learning_for_reliability_analysis/QSAR/model_weihts/'

#print sys.argv
splits = np.arange(0,1)

x_raw=pd.read_csv('/home/peng/Documents/Project_C/QSAR_nlp/qm7_training.csv', header=0)
x_raw= np.array(x_raw)
x_raw =  x_raw[:,1:]
#print x_raw.describe()	




mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

hidden_size = 300

dataset = scipy.io.loadmat('/home/peng/Documents/Project_C/QSAR_nlp/qm7.mat')
for split in splits:
	print "this is the %d split"%split

	# --------------------------------------------
	# Extract training data
	# --------------------------------------------
	P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
	
	x_train = x_raw
		
	x_label = dataset['T'][0,P]
	
	print np.shape(x_label)
	
 
 	# --------------------------------------------
 	### Convert it to 1D for irnn
 	#x_train = x_train.reshape(x_train.shape[0], -1)
 	
 	
 	
 	#x_train = x_train.astype('float32')
 	x_label = x_label.astype('float32')
 	#x_train = preprocessing.normalize(x_train, norm = 'l2')
 	#print (x_train)
 	
 	 #==========================================================================
 	 # --------------------------------------------
 	 # Design the DNN model
 	 # --------------------------------------------
 	 #==========================================================================
  
  
  	model = Sequential()
  	model.add(Dense(output_dim=hidden_size,
  	                    #init=lambda shape, name: normal(shape, scale=0.001, name=name),
  	                    #inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
  	                    activation='linear',
  	                    input_shape=x_train.shape[1:]))
 # 	model.add(Dense(hidden_size, activation='relu'))
  	model.add(Dense(hidden_size, activation='tanh'))
  	model.add(Dense(1, activation='linear'))
  	model.compile(sgd(lr=0.05), "mse")
  
  	model.fit(x_train, x_label, batch_size=mb, nb_epoch=15
  	        )
      ###design DNN and fit
  
  
  	###save the weights
  
  	model.save_weights(root_folder_path + 'nn_keras_%d'%split + ".h5", overwrite=True)	
  	with open(root_folder_path + 'nn_keras_%d'%split + ".json", "w") as outfile:		
  	    json.dump(model.to_json(), outfile)
  
stop = timeit.default_timer()
 
print ("The running takes %r min" %((stop-start)/60))




















#===============================================================================
# 
# 	
# 	# --------------------------------------------
# 	# Create a neural network
# 	# --------------------------------------------
# 	I,O = nn.Input(X),nn.Output(T)
# 	nnsgd = nn.Sequential([I,nn.Linear(I.nbout,400),nn.Sigmoid(),nn.Linear(400,100),nn.Sigmoid(),nn.Linear(100,O.nbinp),O])
# 	nnsgd.modules[-2].W *= 0
# 	nnavg = copy.deepcopy(nnsgd)
# 	
# 	# --------------------------------------------
# 	# Train the neural network
# 	# --------------------------------------------
# 	for i in range(1,1000001):
# 	
# 		if i > 0:     lr = 0.001  # learning rate
# 		if i > 500:   lr = 0.0025
# 		if i > 2500:  lr = 0.005
# 		if i > 12500: lr = 0.01
# 	
# 		r = numpy.random.randint(0,len(X),[mb])
# 		Y = nnsgd.forward(X[r])
# 		nnsgd.backward(Y-T[r])
# 		nnsgd.update(lr)
# 		nnavg.average(nnsgd,(1/hist)/((1/hist)+i))
# 		nnavg.nbiter = i
# 	
# 		if i % 100 == 0: pickle.dump(nnavg,open('nnkeras-%d.pkl'%split,'w'),pickle.HIGHEST_PROTOCOL)
#===============================================================================

