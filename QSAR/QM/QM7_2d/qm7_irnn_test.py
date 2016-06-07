import os,pickle,sys,numpy,copy,scipy,scipy.io
import numpy as np
import json
from keras.models import model_from_json
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


root_folder_path = '/home/peng/Desktop/ Pynotebooks/Link to Machine_learning_for_reliability_analysis/QSAR/model_weihts/'
# --------------------------------------------
# Parameters
# --------------------------------------------
#split = int(sys.argv[1]) # test split for cross-validation (between 0 and 5)
splits = np.arange(0,1)
# --------------------------------------------
# Load data and models
# --------------------------------------------
#if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('/home/peng/Documents/Project_C/QSAR_nlp/qm7.mat')
acc_mae = 0.0

for split in splits:
	with open('/home/peng/Desktop/ Pynotebooks/Link to Machine_learning_for_reliability_analysis/QSAR/model_weihts/nn_keras%d'%split + ".json", 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights('/home/peng/Desktop/ Pynotebooks/Link to Machine_learning_for_reliability_analysis/QSAR/model_weihts/nn_keras%d'%split + '.h5')
	model.compile("sgd", "mse")
	
#	print('results after %d iterations'%nn.nbiter)
	
	Ptrain = dataset['P'][range(0,split)+range(split+1,5)].flatten()
	#Ptest  = dataset['P'][split]
	Ptest  = dataset['P'][split-1]
	for P,name in zip([Ptest],['test']):
		# --------------------------------------------
		# Extract test data
		# --------------------------------------------
		x_train = dataset['X'][P]
		x_train = x_train.reshape(x_train.shape[0], -1, 1)
		x_label = dataset['T'][0,P]
	
		# --------------------------------------------
		# Test the neural network
		# --------------------------------------------
		print('\n%s set:'%name)
		predict_label = model.predict(x_train)
		
		
		
		print('MAE:  %5.2f kcal/mol'%numpy.abs(predict_label-x_label).mean(axis=0))
		acc_mae = acc_mae + numpy.abs(predict_label-x_label).mean(axis=0)
		print('RMSE: %5.2f kcal/mol'%numpy.square(predict_label-x_label).mean(axis=0)**.5)
		
print ' the average_mae is ', acc_mae/5    
