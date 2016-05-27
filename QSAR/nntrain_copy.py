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

# --------------------------------------------
# Parameters
# --------------------------------------------
seed  = 3453
#print sys.argv
splits = np.arange(0,5)

mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

hidden_size = 300

dataset = scipy.io.loadmat('/home/peng/Documents/Project_C/QSAR_nlp/qm7.mat')
for split in splits:
	print "this is the %d split"%split
 # test split



# --------------------------------------------
# Load data
# --------------------------------------------
#numpy.random.seed(seed)
#if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')


	# --------------------------------------------
	# Extract training data
	# --------------------------------------------
	P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
	
	X = dataset['X'][P]
		
	T = dataset['T'][0,P]
	
	
	# --------------------------------------------
	# Design the DNN model
	# --------------------------------------------
	
	model = Sequential()
	model.add(Convolution2D(23,3,3, border_mode='same',
                        input_shape=(3, 23, 23)))
	model.add(Flatten())
			
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(sgd(lr=0.05), "mse")

	model.fit(X, T, batch_size=mb, nb_epoch=15
	        )

	
	
	model.save_weights('/home/peng/Documents/Project_C/QSAR_nlp/nn_keras%d'%split + ".h5", overwrite=True)
	#===========================================================================
	# with open()
	# 
	# with open('/home/peng/Documents/Project_C/QSAR_nlp/nn_keras%d'%split + ".json", "w") as outfile:
	#     json.dump(model.to_json(), outfile)
	#===========================================================================






















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

