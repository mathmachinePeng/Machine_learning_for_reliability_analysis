'''
Created on 28 Jan 2016

@author: peng
'''
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

x= tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

#y_ is the loss function cross entropy
y_ = tf.placeholder("float", [None,10])
#y_* is the real distribution and y is the predicted distribution, measureing the 
# 

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#the optimizing methods can be customized

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#initialize all the variables

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  