import os,pickle,sys,numpy,copy,scipy,scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
split = int(sys.argv[1]) # test split for cross-validation (between 0 and 5)

# --------------------------------------------
# Load data and models
# --------------------------------------------
if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')
nn = pickle.load(open('nn-%d.pkl'%split,'r'))

print('results after %d iterations'%nn.nbiter)

Ptrain = dataset['P'][range(0,split)+range(split+1,5)].flatten()
Ptest  = dataset['P'][split]

for P,name in zip([Ptrain,Ptest],['training','test']):
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
	print('RMSE: %5.2f kcal/mol'%numpy.square(Y-T).mean(axis=0)**.5)

