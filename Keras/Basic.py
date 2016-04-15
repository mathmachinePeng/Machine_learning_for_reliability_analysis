'''
Created on 13 Apr 2016

@author: peng
'''

from keras.models import Sequential

model = Sequential()

from keras.layers.core import Dense, Activation

#Construct the graph network, the input, output nerouns, activation function
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

#And also introduce the loss function(regression or classification), optimizer, metrics
from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#Training, use batch_size to do batch training, and set the number of epochs

"""
Batch Size
Batch size determines how many examples you look at before making a weight update. 
The lower it is, the noisier the training signal is going to be, the higher it is, 
the longer it will take to compute the gradient for each step.
"""

model.fit(X_train=[], Y_train=[], nb_epoch=5, batch_size=32)

# Evaluate it

loss_and_metrics = model.evaluate(X_test=[], Y_test=[], batch_size=32)

# generate the predictions or probabilities of predictions

classes = model.predict_classes(X_test=[], batch_size=32)
proba = model.predict_proba(X_test=[], batch_size=32)

