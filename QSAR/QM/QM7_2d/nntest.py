import os,pickle,sys,numpy,copy,scipy,scipy.io
import numpy as np
# --------------------------------------------
# Parameters
# --------------------------------------------
#split = int(sys.argv[1]) # test split for cross-validation (between 0 and 5)
splits = np.arange(0,5)
# --------------------------------------------
# Load data and models
# --------------------------------------------
#if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('/home/peng/Documents/Project_C/QSAR_nlp/qm7.mat')
acc_mae = 0.0

for split in splits:
	print 

	nn = pickle.load(open('nn-%d.pkl'%split,'r'))
	
	print('results after %d iterations'%nn.nbiter)
	
	Ptrain = dataset['P'][range(0,split)+range(split+1,5)].flatten()
	#Ptest  = dataset['P'][split]
	Ptest  = dataset['P'][split-1]
	for P,name in zip([Ptest],['test']):
		# --------------------------------------------
		# Extract test data
		# --------------------------------------------
		X = dataset['X'][P]
		T = dataset['T'][0,P]
	
		# --------------------------------------------
		# Test the neural network
		# --------------------------------------------
		print('\n%s set:'%name)
		Y = numpy.array([nn.forward(X) for _ in range(10)]).mean(axis=0)
		print('MAE:  %5.2f kcal/mol'%numpy.abs(Y-T).mean(axis=0))
		acc_mae = acc_mae + numpy.abs(Y-T).mean(axis=0)
		print('RMSE: %5.2f kcal/mol'%numpy.square(Y-T).mean(axis=0)**.5)
		
print ' the average_mae is ', acc_mae/5    
