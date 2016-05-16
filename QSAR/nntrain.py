import os,pickle,sys,numpy,nn,copy,scipy,scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
seed  = 3453
#print sys.argv
split = int(sys.argv[1]) # test split

mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

# --------------------------------------------
# Load data
# --------------------------------------------
#numpy.random.seed(seed)
#if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('/home/peng/Documents/Projects/QSAR/Data_source/qm7.mat')

# --------------------------------------------
# Extract training data
# --------------------------------------------
P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
X = dataset['X'][P]
T = dataset['T'][0,P]

# --------------------------------------------
# Create a neural network
# --------------------------------------------
I,O = nn.Input(X),nn.Output(T)
nnsgd = nn.Sequential([I,nn.Linear(I.nbout,400),nn.Sigmoid(),nn.Linear(400,100),nn.Sigmoid(),nn.Linear(100,O.nbinp),O])
nnsgd.modules[-2].W *= 0
nnavg = copy.deepcopy(nnsgd)

# --------------------------------------------
# Train the neural network
# --------------------------------------------
for i in range(1,1000001):

	if i > 0:     lr = 0.001  # learning rate
	if i > 500:   lr = 0.0025
	if i > 2500:  lr = 0.005
	if i > 12500: lr = 0.01

	r = numpy.random.randint(0,len(X),[mb])
	Y = nnsgd.forward(X[r])
	nnsgd.backward(Y-T[r])
	nnsgd.update(lr)
	nnavg.average(nnsgd,(1/hist)/((1/hist)+i))
	nnavg.nbiter = i

	if i % 100 == 0: pickle.dump(nnavg,open('nn-%d.pkl'%split,'w'),pickle.HIGHEST_PROTOCOL)

